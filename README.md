# why-cite
**Investigating the purpose of citations in scientific literature.**

### How to run
- Install all requirements by running the command ```pip3 install -r requirements.txt```. 
- Setup the config of your choice in ```src/configs/config.yaml```
  - In losses, set the weights of weighted cross entropy
  - In training, set the learning rate, batch sizes, model needed, and dataset desired (Shared Task 3, ACL-ARC, SciCite)
  - the model is set up with wandb in order to track the scores of the model as it trains
- In order to train the model run ```./training_scripts/sample.sh```.

