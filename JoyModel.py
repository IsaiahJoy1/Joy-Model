import copy
import h5py
import json
import logging
import matplotlib
import operator
import os
import pathlib
import pickle
import shutil
import absl.logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, GRU, Input, Concatenate, Layer
from keras.models import Model
from math import ceil
from multimethod import multimethod
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)
matplotlib.interactive(False)

class Mapping(dict):

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

#------------------ Data Sructs ------------------

@dataclass_json
@dataclass
class DataNote(Mapping):
    name:str = field(default_factory=str)
    data_type: str = field(default_factory=str)
    padding: int|None = None

@dataclass_json
@dataclass
class DataStructFromat(Mapping):
    id : any = field(default_factory=str)
    label: any = 0.0
    data: dict[list] = field(default_factory=dict)

    def __post_init__(self):
        self.__dict__.update(self.__dict__.pop("data")) 

@dataclass_json
@dataclass
class MLDataStruct(Mapping):
    data : list[DataStructFromat] = field(default_factory=list)
    notes : list[DataNote] = field(default_factory=list)

    def __post_init__(self):
        if len(self.data) and not isinstance(self.data[0], DataStructFromat):
            self.data = [DataStructFromat(kwargs.pop("id"), kwargs.pop("label"), kwargs) for kwargs in self.data]

        if len(self.notes) and not isinstance(self.notes[0], DataNote):
            self.notes = [DataNote(**kwargs) for kwargs in self.notes]

@dataclass_json
@dataclass
class DataSet(Mapping):
    id   : list = field(default_factory=list)
    data : list[list] = field(default_factory=list)
    label: list = field(default_factory=list)

@dataclass_json
@dataclass
class TrainingSets(Mapping):
    train: DataSet = field(default_factory=DataSet)
    test : DataSet = field(default_factory=DataSet)
    val  : DataSet = field(default_factory=DataSet)

#------------------ Model Sructs -----------------

@dataclass_json
@dataclass
class ModelBranch(Mapping):
    name: str = field(default_factory=str)
    data_type: str = field(default_factory=str)
    layers: list[dict[str,callable]]  = field(default_factory=list)

    def __post_init__(self):
        if len(self.layers):
            for key, input in self.layers[0].items():
                if isinstance(input, list):
                    self.layers[0][key] = tuple(input)

    def operate_on_branch(self, num:int|float, operation:operator, safe_layer:int) -> None:
            for layer in range(len(self.layers)):
                if layer != safe_layer:
                    for node in list(self.layers[layer].keys()):
                        self.layers[layer][node] = int(operation(self.layers[layer][node], num))

@dataclass_json
@dataclass
class ModelBlueprint(Mapping):
    inputs: list[ModelBranch] = field(default_factory=list)
    output: ModelBranch = field(default_factory=ModelBranch)

    def __post_init__(self):
        if len(self.inputs) and not isinstance(self.inputs[0], ModelBranch):
            self.inputs = [ModelBranch(**kwargs) for kwargs in self.inputs]
        
        if not isinstance(self.output, ModelBranch):
            self.output = ModelBranch(**self.output)

    def _oper_per_branch(self, num:int|float, operation:operator) -> None:
        for branch in self.inputs:
            branch.operate_on_branch(num, operation,0)

        self.output.operate_on_branch(num, operation, len(branch) -1)

    def __add__(self, val:int|float):
        self._oper_per_branch(val, operator.add)

    def __mul__(self, scaler:int|float) -> None:
        self._oper_per_branch(scaler, operator.mul)

@dataclass_json
@dataclass
class MLSave(Mapping):
    name : str
    loss_function : str
    func_optimizer : str
    blueprint : ModelBlueprint = None
    loss : float = None

    def __post_init__(self):
        if self.blueprint is not None and not isinstance(self.blueprint, ModelBlueprint):
            self.blueprint = ModelBlueprint(**self.blueprint)

#------------------ Model Classes ----------------

class ActivationFuncMaper:
    Dense_acts : dict
    RNN_layers : dict

    def __init__(self, Dense:dict[str, str|Callable]=None, RNN:dict[str, str|Callable]=None, clear_defaults:bool=False):
        """
        Dense: A dict that maps str names to Dense Activation Functions\n
        RNN : A dict that maps str names to RNN layers\n
        Defaults:\n
        \tDense: tanh, leaky_relu, linear\n
        \tRNN: lstm, gru
        """
        if Dense is None:
            Dense = {}
        if RNN is None:
            RNN = {}

        if clear_defaults:
            self.Dense_acts = {}
            self.RNN_layers = {}
        else:
            # Activation Function that Cross the Origin
            self.Dense_acts = {'tanh':'tanh', 'leaky_relu':'leaky_relu', 'linear':'linear'}
            # GPU Supported RNNS
            self.RNN_layers = {'lstm':LSTM, 'gru':GRU}

        self.Dense_acts.update(Dense)
        self.RNN_layers.update(RNN)

    def Dense_Keys(self) -> list:
        return list(self.Dense_acts.keys())
    
    def RNN_Keys(self) -> list:
        return list(self.RNN_layers.keys())
    
    def RNN_Layer_Types(self) -> list:
        temp = self.Dense_Keys()
        temp.extend(self.RNN_Keys())
        return temp

    def make_layer(self, layer:str, width:int|tuple, return_sequences:bool, name:str, previous_layer:Layer) -> Layer:
        if layer in self.RNN_Keys():
            return self.RNN_layers[layer](width, return_sequences=return_sequences, name=name)(previous_layer)
        elif layer == 'inputlayer':
            return Input(shape=width, name=name)
        else:
            return Dense(width, activation = self.Dense_acts[layer], name=name)(previous_layer)
            
class MLModel:
    # files
    model : MLSave
    maper : ActivationFuncMaper
    folder : str
    # model
    device : str

    def __init__(self, saved_model:MLSave, maper:ActivationFuncMaper, 
                 folder:str="./model_saves", device:str ='/CPU:0') -> None:
        
        self.model = saved_model
        self.maper = maper
        if not(folder[-1] != "/" or folder[-1] != "\\"):
            folder = folder + "/"
        self.folder = folder
        self.device = device

    def _get_chpt_location(self, model:MLSave) -> str:
        return f"{self.folder}{model.name}/ckpt.h5"

    def _compile_saved_model(self, model:MLSave=None,optimizer=None) -> Model:
        if model is None:
            model = self.model
        cmodel = compile_model(model, self.maper, optimizer)
        cmodel.load_weights(self._get_chpt_location(model))
        return cmodel

    #---------------------------------------------------------
    #               RUN IN A SEPARATE PROCESS
    #closing a process will recover gpu memory after a failure  

    def _copy(self, model:MLSave):
        """RUN IN A SEPARATE PROCESS"""
        timp_model = self._compile_saved_model()
        create_h5_file(self._get_chpt_location(model))
        timp_model.save_weights(self._get_chpt_location(model))
        copy_MLSave(self.model, model)
    
    def _eval_model(self, predict:DataSet, model:Model) -> float|int:
        """
        RUN IN A SEPARATE PROCESS
        returns -1 if all elements are the same
        """
        predicted = round_list(model(predict["data"]).numpy().tolist(), 4)
        pred_libel = round_list(predict['label'].tolist(), 4)
        if len(set(tuple(lst) for lst in predicted)) == 1 and len(set(tuple(lst) for lst in pred_libel)) != 1:
            loss = -1
        else:
            loss = model.evaluate(predict["data"],predict["label"])
            if isinstance(loss, list):
                loss = loss[0]
        return loss
    
    def _predict(self, data:DataSet) -> int|list:
        """RUN IN A SEPARATE PROCESS"""
        with tf.device(self.device):
            try:
                model = self._compile_saved_model()
                predict = model(data["data"]).numpy().tolist()
            except Exception:
                predict = -1
            return predict

    def _test(self, data:DataSet) -> float|int:
        """RUN IN A SEPARATE PROCESS"""
        with tf.device(self.device):
            try:
                model = self._compile_saved_model()
                predict = self._eval_model(data, model)
            except Exception:
                predict = -1
            return predict

    def _png(self, *args):
        """RUN IN A SEPARATE PROCESS"""
        model = compile_model(self.model, self.maper)
        tf.keras.utils.plot_model(model, to_file=f"{self.folder}{self.model.name}.png",show_shapes=True)

    #---------------------------------------------------------

    def predict(self, data:DataSet) -> list|int:
        with Pool(1) as p:
            predict = p.starmap(self._predict,[(data,)])
        return predict[0]
        
    def test(self, data:DataSet) -> float|int:
        with Pool(1) as p:
            test = p.starmap(self._test, [(data,)])
        return test[0]

    def copy_MLSave(self, model:MLSave):
        with Pool(1) as p:
            ret = p.starmap(self._copy, [(model,)])
        copy_MLSave(self.model, model)

    def make_png(self):
        with Pool(1) as p:
            ret = p.map(self._png, [1])

class ML_Model_Trainer(MLModel):
    # objs
    data : TrainingSets
    best_model : MLSave
    # fitting / pruning
    early_stoping_patience : int
    callback_list : list
    # fitting
    # func_optimizer default list: ftrl, adadelta, adam, adagrad, adamax, nadam, rmsprop, sgd
    GPU : str
    batch_size : int
    max_number_of_epochs : int

    def __init__(self, train_model:MLSave, best_model:MLSave, maper:ActivationFuncMaper, data:TrainingSets, 
                 folder:str="./model_saves/", GPU:str='/device:GPU:0',
                 max_number_of_epochs:int=10000, batch_size:float=.1, 
                 early_stoping_patience:int=5, min_delta:float=0.0005) -> None:
        """
        train_model: A MLSave used to build and train your model
        best_model: A check point of your best trained model
        """
        
        MLModel.__init__(self, train_model, maper, folder, GPU)
        self.best_model = best_model
        self.data = data
        self.GPU = GPU
        self.max_number_of_epochs = max_number_of_epochs
        self.batch_size = batch_size
        self.early_stoping_patience = early_stoping_patience
        self.callback_list = self.make_callback_list(early_stoping_patience, min_delta)
    
    #---------------------------------------------------------
    #           functions to reduce repeated code
    
    def _calculate_batch_size(self, size:int|float = None) -> int:
        if size == None:
            size = self.batch_size
        if size < 1:
            size = size * len(self.data.train['data'][-1])
            if size < 1:
                size = 1
        return int(size)

    def make_callback_list(self, early_stoping_patience:int=5 , min_delta:float=0.0005)->list:
        monitor = 'val_loss'
        checkpoint = ModelCheckpoint(self._get_chpt_location(self.model), monitor = monitor, 
                                     verbose = 0, save_best_only=True, mode = 'min') #, save_weights_only=True
        early_stoping = EarlyStopping(monitor=monitor, mode='min', verbose=0, patience=early_stoping_patience, 
                                      min_delta=min_delta)
        callback_list = [checkpoint,early_stoping]
        return callback_list

    #---------------------------------------------------------

    #---------------------------------------------------------
    #               RUN IN A SEPARATE PROCESS
    #closing a process will recover gpu memory after a failure  

    def _save_best(self):
        """RUN IN A SEPARATE PROCESS"""
        if (self.model.loss is not None and
                    self.model.loss != -1 and
                    (self.best_model.loss is None or 
                     self.best_model.loss > self.model.loss)):
            self._copy(self.best_model)

    def _fit(self, batch_size:int, verbose:int=0) -> tuple[dict|int, MLSave, MLSave]:
        """RUN IN A SEPARATE PROCESS"""
        try:
            model = compile_model(self.model, self.maper)
        except Exception:
            return -1
        if verbose > 1:
             model.summary()
             verbose = 1   
        with tf.device(self.device):
            try:
                history = model.fit(self.data.train["data"], self.data.train["label"], epochs = self.max_number_of_epochs, batch_size = batch_size,
                            validation_data=(self.data.val["data"], self.data.val["label"]), verbose = verbose, callbacks = self.callback_list)
                history = history.history
                model.load_weights(self._get_chpt_location(self.model))
                self.model.loss = self._eval_model(self.data.val, model)
                self._save_best()
                if len(history['val_loss']) < 2*self.early_stoping_patience:
                    history = -2
            except Exception as e:
                logging.error('Failed to upload to ftp: '+ str(e))
                history = -1

            return history, self.model, self.best_model
        
    def _refit(self, func_optimizer:str|None, verbose:int = 0) -> tuple[dict|int, MLSave, MLSave]:
        """RUN IN A SEPARATE PROCESS"""
        with tf.device(self.device):
            batch_size = self._calculate_batch_size(self.batch_size)
            try:
                model = self._compile_saved_model(optimizer=func_optimizer)
            except Exception:
                return -1
            if verbose > 1:
                model.summary()
                verbose = 1
            try:
                history = model.fit(self.data.train["data"], self.data.train["label"], epochs = self.max_number_of_epochs, batch_size = batch_size,
                            validation_data=(self.data.val["data"], self.data.val["label"]), verbose = verbose, callbacks = self.callback_list)
                history = history.history
                model.load_weights(self._get_chpt_location(self.model))
                self.model.loss = self._eval_model(self.data.val, model)
                self._save_best()
                if len(history['val_loss']) < 2*self.early_stoping_patience:
                    history = -2
            except Exception as e:
                logging.error('Failed to upload to ftp: '+ str(e))
                history = -1
            
            return history, self.model, self.best_model

    #---------------------------------------------------------

    #---------------------------------------------------------
    #       Basice functions to that safely use the model

    def _map_unpacker(self, history, model, best_model):
        copy_MLSave(model, self.model)
        copy_MLSave(best_model, self.best_model)
        return history

    def train(self, batch_size:int|float=None, skipException:bool = True, verbose:int=0):
        """
        skipException: if model fails training on GPU then try on CPU:0
        """
        batch_size = self._calculate_batch_size(batch_size)
        self.device = self.GPU
        
        with Pool(1) as p:
            history = p.starmap(self._fit, [(batch_size, verbose)])
        history = self._map_unpacker(*history[0])

        if history == -1:
            if skipException:
                print("GPU Error, Skipping\n")
            else:
                print("GPU Error, running on CPU\n")
                self.device = '/CPU:0'
                with Pool(1) as p:
                    history = p.starmap(self._fit, [(batch_size, verbose)])
                history = self._map_unpacker(*history[0])

                if history == -1:
                    print("Training Error\n")
        return history

    def retrain(self, func_optimizer:str, skipException:bool = True, verbose:int=0):
        """
        skipException: if model fails training on GPU then try on CPU:0
        """
        self.device = self.GPU
        with Pool(1) as p:
            history = p.starmap(self._refit, [(func_optimizer,verbose)])
        history = self._map_unpacker(*history[0])

        if history == -1:
            with Pool(1) as p:
                history = p.starmap(self._refit, [(func_optimizer,verbose)])
            history = self._map_unpacker(*history[0])

        if history == -1:
            if skipException:
                print("GPU Error, Skipping\n")
            else:
                print("GPU Error, running on CPU\n")
                self.device = '/CPU:0'
                with Pool(1) as p:
                    history = p.starmap(self._refit, [(func_optimizer,verbose)])
                history = self._map_unpacker(*history[0])

                if isinstance(history, int):
                    print("Training Error\n")
        return history  
    #---------------------------------------------------------
    
    #---------------------------------------------------------
    #      functions that loop over standared functions
    
    def train_loop(self, batch_size:int|float=None, batch_growth_rate:float = .75, skipException:bool = True, verbose=0):
        """
        batch_growth_rate: if traing model fails set batch size to len(train set)*.5*rate^(iteration)
        skipException: if model fails training on GPU then try on CPU:0
        """
        history = self.train(batch_size=batch_size, skipException = skipException, verbose=verbose)

        if not isinstance(history, int):
            loss = self.model.loss
            if verbose >= 1:
                print(loss)
        else:
            loss = -1

        if loss == -1:
            i = 0
            size_of_batch = .5
            if verbose > 1:
                verbose =1

        while (loss == -1) and (pow(batch_growth_rate,i) *size_of_batch < 1) and size_of_batch*pow(batch_growth_rate,i)*len(self.data.train["data"][-1]) > 1:
            history = self.train(batch_size=size_of_batch*pow(batch_growth_rate,i), skipException = skipException, verbose=verbose)
            if not isinstance(history, int):
                loss = self.model.loss
                if verbose == 1:
                    print(loss)
            i = i+1
            
        return history, loss

    def polt_train(self):
        history, loss= self.train_loop(verbose=2)

        if not isinstance(history, int):
            training_loss = history['loss']
            validation_loss = history['val_loss']

            # Create count of the number of epochs
            epoch_count = range(1, len(training_loss) + 1)

            # Visualize loss history
            plt.plot(epoch_count, training_loss, 'r--')
            plt.plot(epoch_count, validation_loss, 'b-')
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        return loss

    def del_weights(self):
        delete_folder(f"{self.folder}{self.model.name}")

    #---------------------------------------------------------

class MLProject:
    project_path:str
    name:str

    _train_data:TrainingSets
    _train_data_file="data.pickle"
    _func_maper:ActivationFuncMaper
    _func_maper_file="maper.pickle"

    saved_models:dict[str, MLSave]
    _saved_models_file="saved_models.json"

    @multimethod
    def __init__(self, name:str, project_path:str="./MLProjects"):
        self.name = name
        self.project_path = project_path

        project_path = self.file_path()
        with open(project_path+self._func_maper_file, "+rb") as file:
            self._func_maper = pickle.load(file)

        with open(project_path+self._train_data_file, '+rb') as file:
            self._train_data = pickle.load(file)

        with open(project_path+self._saved_models_file, 'r') as file:
            self.saved_models = json.load(file)

        for key, model in self.saved_models.items():
            self.saved_models[key] = MLSave(**model)

    @multimethod
    def __init__(self, name:str, data:MLDataStruct, model:MLSave, activation_funcs:ActivationFuncMaper=None, project_path:str="./MLProjects"):
        if activation_funcs is None:
            activation_funcs = ActivationFuncMaper()
        
        self.name = name
        self.project_path = project_path
        self._func_maper = activation_funcs
        self._train_data = data_struct_2_training_sets(data)
        self.saved_models = {}

        if not isinstance(model.blueprint, ModelBlueprint):
            model.blueprint = make_model_blueprint(data, activation_funcs)
        self._make_base_model_copys(model)

        create_folder(self.file_path())

    @multimethod
    def __init__(self, name:str, data:TrainingSets, model:MLSave, activation_funcs:ActivationFuncMaper=None, project_path:str="./MLProjects"):
        if activation_funcs is None:
            activation_funcs = ActivationFuncMaper()

        self.name = name
        self.project_path = project_path
        self._func_maper = activation_funcs
        self._train_data = data
        self.saved_models = {}

        if not isinstance(model.blueprint, ModelBlueprint):
            raise TypeError("model.blueprint must be of type ModelBlueprint")
        self._make_base_model_copys(model)

        create_folder(self.file_path())

    def _make_base_model_copys(self, model:MLSave):
        train = copy.deepcopy(model)
        best = copy.deepcopy(model)
        train.name = "train"
        best.name = "best"
        self.saved_models["save"] = model
        self.saved_models["train"] = train
        self.saved_models["best"] = best

    def file_path(self) -> str:
        return f"{self.project_path}/{self.name}/"

    def save_project(self):
        project_path = self.file_path()
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        abspath = pathlib.Path(project_path+self._func_maper_file).absolute()
        with open(abspath, 'wb') as file:
            pickle.dump(self._func_maper, file)
        
        with open(project_path+self._train_data_file, 'wb') as file:
            pickle.dump(self._train_data, file)

        with open(project_path+self._saved_models_file, 'w') as file:
            json.dump(self.saved_models, file)

    def make_Trainer(self, GPU:str='/device:GPU:0',
                    max_number_of_epochs:int=10000, batch_size:float=.1, 
                    early_stoping_patience:int=5, min_delta:float=0.0005) -> ML_Model_Trainer:
        return ML_Model_Trainer(self.saved_models["train"], self.saved_models["best"],
                                self._func_maper, self._train_data, self.file_path(),
                                GPU, max_number_of_epochs, batch_size, early_stoping_patience, min_delta)

    def make_MLModel(self, name:str=None, device:str = '/CPU:0') -> MLModel:
        if name is None:
            name = "save"
        return MLModel(self.saved_models[name], self._func_maper, self.file_path(), device)

    def save_best(self):
        timp = MLModel(self.saved_models["save"])
        timp.copy_MLSave(self.saved_models["best"])
        
#-------------------- Functions -------------------

def layer_width(width:int|float|tuple) -> int|tuple:
    if isinstance(width, tuple):
        return width
    elif width <= 0:
        return 1
    else:
        return ceil(width)
    
def layer_name(input_name:str, layernum:int, node_type:str) -> str:
    return f"{input_name}_{str(layernum)}_{node_type}"

def make_model_blueprint(data_struct:MLDataStruct, maper:ActivationFuncMaper, outputwidth:int=1, width_scale:int|float = 1) -> ModelBlueprint:
    
    def __make_sub_struct_layer(activation_list:list[str], width:int|float) -> dict:
        return {layer_type:layer_width(width) for layer_type in activation_list}

    def __make_output_sub_struct(input_layers:list[ModelBranch], output_width:int|float) -> list[dict]:
        total_width = 0
        #width is the avg of the input models
        for inputs in input_layers:
            total_width += inputs['layers'][-1][(list(inputs['layers'][-1])[0])]
        width = total_width//len(input_layers)
        sub_struct = [__make_sub_struct_layer(maper.Dense_Keys(), (width+output_width)/2),
                        {'linear': output_width}] 
        return sub_struct

    def __make_sub_struct(data:list[DataStructFromat], subset_data:DataNote, width_scale:int|float = 1) -> ModelBranch:
        sub_struct = ModelBranch(subset_data.name, subset_data.data_type, [])
        match subset_data.data_type:
            case "RNN":
                width = len(data[0][subset_data['name']][0])
                sub_struct.layers.append({'inputlayer':(None,width)})
                sub_struct.layers.extend([__make_sub_struct_layer(maper.RNN_Layer_Types(), width*2*width_scale),
                                            __make_sub_struct_layer(maper.RNN_Keys(), width*width_scale)])
            case "NN":
                width = len(data[0][subset_data['name']])
                sub_struct.layers.append({'inputlayer': width})
                sub_struct.layers.extend([__make_sub_struct_layer(maper.Dense_Keys(), width*2*width_scale),
                                            __make_sub_struct_layer(maper.Dense_Keys(), width*width_scale)])
        return sub_struct

    model_struct = ModelBlueprint([__make_sub_struct(data_struct['data'], subset, width_scale)
                                            for subset in data_struct.notes])
    model_struct.output = ModelBranch("output","NN",__make_output_sub_struct(model_struct.inputs, outputwidth))
    return model_struct

def compile_model(save:MLSave, maper:ActivationFuncMaper, optimizer=None) -> keras.models.Model:
    def __make_layer(input_name:str, layer_nodes:list, layernum:int, previous_layer:Layer, return_sequences:bool) -> Layer:
        layer = [maper.make_layer(i, layer_width(layer_nodes[i]), return_sequences, 
                                layer_name(input_name, layernum, i), previous_layer)
                    for i in layer_nodes]
        if len(layer) > 1:
            return Concatenate()(layer)
        else:
            return layer[0]
        
    struct = save.blueprint
    inputs = [None]*len(struct.inputs)
    inputs_layers = [None]*len(struct.inputs)

    for i in range(len(struct.inputs)):
        previous_layer=None
        layers = struct.inputs[i].layers
        for j in range(len(layers)):
            #not last layer n the input
            return_sequences = j != len(layers)-1
            sub_layer = __make_layer(struct.inputs[i].name,layers[j], j, previous_layer, return_sequences)
            #save inputlayers and last
            if j == 0:
                inputs[i] = sub_layer
            if return_sequences == False:
                inputs_layers[i] = sub_layer
            previous_layer = sub_layer

    if len(inputs_layers) != 1:
        output = Concatenate()(inputs_layers)
    else:
        output = inputs_layers[0]

    for i in range(len(struct.output.layers)):
        output = __make_layer(struct.output.name,struct.output.layers[i], i, output, False)

    model = Model(inputs=inputs, outputs=output)

    if optimizer is None:
        optimizer = save.func_optimizer
    model.compile(loss=save.loss_function, optimizer=optimizer)
    return model

def copy_MLSave(original:MLSave, model:MLSave):
    model_copy = copy.deepcopy(original)
    name = model.name 
    model.__dict__.update(model_copy.__dict__)
    model.name = name


def extractkeysfromlist(dataList:list[dict], dataType:str) -> list:
    data = []
    for index in dataList:
        data.append(index[dataType])
    return data

def round_list(mylist:list,place:int) -> list|float:
    if isinstance(mylist, list):
        return [round_list(num, place) for num in mylist]
    else:
        return round(mylist, place)

def pad_data(data:MLDataStruct) -> None:
    for track in data['data']:
            for data_type in data['notes']:
                if data_type['data_type'] == "RNN":
                    timp = [[0]*len(track[data_type['name']][0])]*(data_type['padding']-len(track[data_type['name']]))
                    timp.extend(track[data_type['name']])
                    track[data_type['name']] = timp

def data_struct_2_training_sets(data:MLDataStruct, train_split=.8, val_split=.5) -> TrainingSets:
    pad_data(data)
    training_sets = TrainingSets({},{},{})
    dataSplits = {}
    dataSplits["train"], dataSplits["test"] = train_test_split(data['data'], train_size=train_split, random_state=0)
    dataSplits["val"], dataSplits["test"] = train_test_split(dataSplits["test"], train_size=val_split, random_state=0)

    for dataset in dataSplits.keys():
        training_sets[dataset]["data"] = []
        for i in data['notes']:
            training_sets[dataset]["data"].append(np.array(extractkeysfromlist(dataSplits[dataset],i['name'])))
        training_sets[dataset]['label'] = np.array(extractkeysfromlist(dataSplits[dataset],'label'))
        training_sets[dataset]['id'] = np.array(extractkeysfromlist(dataSplits[dataset],'id'))

    return training_sets

def data_struct_transform(data:MLDataStruct):
    pad_data(data)
    dataset = DataSet([],[],[])
    for i in data['notes']:
        dataset.data.append(np.array(extractkeysfromlist(data,i['name'])))
    dataset["label"] = np.array(extractkeysfromlist(data,'label'))
    dataset["id"] = np.array(extractkeysfromlist(data,'id'))


def create_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'x'):
            pass

def create_h5_file(file_path):
    folder_path = os.path.dirname(file_path)
    create_folder(folder_path)
    if not os.path.exists(file_path):
        with h5py.File(file_path, 'a') as h5_file:
            pass

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

