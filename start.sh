### some Baysian
# for value in 0 5 35 2 8 33 22 37 4 30
for value in 0.06 0.01 0.02 0.03 0.08 0.1 0.12

do
echo value : $value
# python perturbation.py --target=$value --num_iter=50 --step=1
# python natural.py --target=$value --initial_alpha=0 --step=3 --batch_size=5 --num_iter=50 --lr_attack=0.01 --initial_weight=1 --gamma=0.5 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --bound_ball=0.03 --hard_bound True --hard_upper_bound=0.4
# python natural.py --target=$value --initial_alpha=0 --step=8 --batch_size=5 --num_iter=80 --lr_attack=0.01 --initial_weight=10 --gamma=0.5 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --hard_bound_mode=0 --dyn_bound_mode=2 --b_two=0.8 --b_infty=0.03 --s_infty=0.1 --u_two=3 --u_infty=0.5
python natural.py --phase all --initial_alpha=0 --step=1 --batch_size=5 --num_iter=100 --lr_attack=0.01 --initial_weight=10 --gamma=0 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --hard_bound_mode=1 --dyn_bound_mode=1 --b_two=0.8 --b_infty=$value --s_infty=0.1 --u_two=4 --u_infty=$value
done
echo finished no AE experimtn 


for value in 0.06 0.01 0.02 0.03 0.08 0.1 0.12

do
echo value : $value
python natural.py --phase all --initial_alpha=0 --step=1 --batch_size=5 --num_iter=100 --lr_attack=0.01 --initial_weight=10 --gamma=0.5 --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --hard_bound_mode=1 --dyn_bound_mode=1 --b_two=0.8 --b_infty=$value --s_infty=0.1 --u_two=4 --u_infty=$value
done
echo finished with AE experimtn 