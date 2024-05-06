Most of the project code is in the python source files.
The Jupyter notebooks are just wrappers are the python files for running in Google colab
Refer to the following table below:

Name                    | Direct Call
--------------------------------------------------------------------------------
FlirDataset.py          | Plots Bounding Boxes on Input Image
NetworkTrainer.py       | Train PyTorch Built-In Faster RCNN
fasterRCNN.py           | Train Custom Faster RCNN
GetRegionProposals.py   | Plot output of region proposal network
runNetwork.py           | Display Training Results for Custom Faster RCNN
runBuiltInNetwork.py    | Display Training Resutls for Built-In Faster RCNN
ComputeMetrics.py       | Computes mAP, mAR, IOU and plots outputs

The Following Jupyter Notebooks are wrappers around the python files

Name                            | Direct Call
--------------------------------------------------------------------------------
TrainDefaultFasterRCNN.ipynb    | Train Default Faster RCNN
TrainFasterRCNN.ipynb           | Train Custom Faster RCNN
Test_Network_Metrics.ipynb      | Generate Metrics

NOTE:   Run data will be created in ../runs
        FLIR_ADAS_v2 dataset will be copied to ~/FLIR_ADAS_v2
