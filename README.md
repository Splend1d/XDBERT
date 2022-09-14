# XDBERT
Code for ACL 2022 Conference Paper "XDBERT: Distilling Visual Information to BERT from Cross-Modal Systems to Improve Language Understanding"


## Pretraining
preparing data : download wiki to data/wiki-cased with 
```bash
kaggle datasets download -d a24998667/wiki-raw
``````
Then run,
```
bash xattn_pretrain.sh
```


## Finetuning
The glue dataset is downloadable using the datasets lib, so no extra preparation is needed 

Simply run 
```
bash distil_run_glue.sh
```
with the pretrained checkpoint