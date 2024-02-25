python runattack.py --data_dir ./data/dataset/raw/Dataset --labels T-shirt/Top,Hosen,Pullover,Kleid,Mantel,Sandalen,Shirt,Sneaker,Rucksack,Ankle_boot --requestURL http://127.0.0.1:5000/cnn/decision --requestHandlerClass ImageDecision --refname ID01 --attack_name HSJ --metric linf --params_bool targeted:0,verbose:0 --params_int iter_max:[3],eval_max:[10;100;200],eval_init:[5],iter:[5]

python runattack.py --data_dir ./data/dataset/raw/Dataset --labels T-shirt/Top,Hosen,Pullover,Kleid,Mantel,Sandalen,Shirt,Sneaker,Rucksack,Ankle_boot --requestURL http://127.0.0.1:5000/cnn/decision --requestHandlerClass ImageDecision --refname ID02 --attack_name SignOpt --params_bool targeted:0,verbose:0 --params_int iter_max:[5,7],num_trial:[2],query_limit:[50],k:[2],eval_perform:[1] --params_float epsilon:[0.001],alpha:[0.2],beta:[0.001]


