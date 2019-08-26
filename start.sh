### some Baysian
for value in 0 5 35 2 8 33 22 37 4 30
do
echo exp : $value
# python perturbation.py --target=$value --num_iter=50 --step=1
# python natural.py --target=$value --initial_alpha=0 --step=3 --batch_size=5 --num_iter=50 --lr_attack=0.01 --initial_weight=1 --gamma=0.5 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --bound_ball=0.03 --hard_bound True --hard_upper_bound=0.4
python natural.py --target=$value --initial_alpha=0 --step=8 --batch_size=5 --num_iter=80 --lr_attack=0.01 --initial_weight=10 --gamma=0.5 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=1 --bound_ball=0.03 --hard_upper_bound=0.4
done
echo finished training 