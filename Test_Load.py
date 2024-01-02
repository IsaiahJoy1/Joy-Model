import JoyModel as jm

if __name__ == '__main__':
    project = jm.MLProject("spotify")
    model = project.make_Trainer()
    model.retrain("adam",verbose = 1)
    project.save_project()
    