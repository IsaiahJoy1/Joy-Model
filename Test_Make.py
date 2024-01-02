import json
import JoyModel as jm

if __name__ == '__main__':
    with open('data1.json', 'r') as fp:
        my_data = json.load(fp)
 
    data = jm.MLDataStruct(**my_data)
    #fun_optimizer = [ftrl, adadelta, adam, adagrad, adamax, nadam, rmsprop, sgd]
    project = jm.MLProject("spotify", data, 
                             jm.MLSave("spotify", "mean_absolute_error", "adagrad"),
                             jm.ActivationFuncMaper())
    
    model = project.make_Trainer()
    model.del_weights()
    model.make_png()
    model.polt_train()
    model.retrain("adam", verbose = 1)
    project.save_project()