import seaborn
accuracies = {
    "orig_acc":orig_acc  ,
  "adv_suc":adv_suc  ,
  "adv_acc":adv_acc,
   "proj_acc":proj_acc  ,
   "rec_suc":rec_suc  ,
   "rec_acc":rec_acc,
   "orig_acc_p":orig_acc_p  ,
    "adv_suc_p":adv_suc_p  ,
  "adv_acc_p":adv_acc_p,
   "proj_acc_p":proj_acc_p  ,
   "rec_suc_p":rec_suc_p  ,
   "rec_acc_p":rec_acc_p,
  "b_adv_suc":b_adv_suc,
  "b_adv_acc":b_adv_acc,
  "b_rec_suc":b_rec_suc,
  "b_rec_acc":b_rec_acc,
  "b_adv_suc_p":b_adv_suc_p,
  "b_adv_acc_p":b_adv_acc_p,
  "b_rec_suc_p":b_rec_suc_p,
  "b_rec_acc_p":b_rec_acc_p,
    "orig_acc_r":orig_acc_r,
  "adv_suc_r":adv_suc_r,
  "adv_acc_r":adv_acc_r,
  "b_adv_suc_r":b_adv_suc_r,
  "b_adv_acc_r":b_adv_acc_r  ,
  "orig_acc_o":orig_acc_o,
  "adv_suc_o":adv_suc_o,
  "adv_acc_o":adv_acc_o,
  "b_adv_suc_o":b_adv_suc_o,
  "b_adv_acc_o":b_adv_acc_o  }
accuracies_disc = {
    "orig_acc": "original accuracy on PointNet ",
    "adv_suc": "natural adverserial sucess rate on PointNet ",
    "adv_acc": "natural adverserial accuracy on PointNet ",
    "proj_acc": "projected accuracy on PointNet ",
    "rec_suc": "defended natural adverserial sucess rate on PointNet ",
    "rec_acc": "reconstructed defense accuracy on PointNet ",
    "orig_acc_p": "original accuracy on PointNet_++ ",
    "adv_suc_p": "natural adverserial sucess rate on PointNet_++ ",
    "adv_acc_p": "natural adverserial accuracy on PointNet_++ ",
    "proj_acc_p": "projected accuracy on PointNet_++ ",
    "rec_suc_p": "defended natural adverserial sucess rate on PointNet_++ ",
    "rec_acc_p": "reconstructed defense accuracy on PointNet_++ ",
    "b_adv_suc": "baseline adverserial sucess rate on PointNet ",
    "b_adv_acc": "baseline adverserial accuracy on PointNet ",
    "b_rec_suc": "baseline defended natural adverserial sucess rate on PointNet ",
    "b_rec_acc": "baselin ereconstructed defense accuracy on PointNet ",
    "b_adv_suc_p": "baseline adverserial sucess rate on PointNet_++ ",
    "b_adv_acc_p": "baseline adverserial accuracy on PointNet_++ ",
    "b_rec_suc_p": "baseline defended natural adverserial sucess rate on PointNet_++ ",
    "b_rec_acc_p": "baselin ereconstructed defense accuracy on PointNet_++ ",
    "orig_acc_r": "original accuracy under Random defense",
    "adv_suc_r": "natural adverserial accuracy under Random defense",
    "adv_acc_r": "natural adverserial sucess rate under Random defense",
    "b_adv_suc_r": "baseline  adverserial accuracy under Random defense",
    "b_adv_acc_r": "baseline  adverserial sucess rate under Random defense",
    "orig_acc_o": "original accuracy under Outlier defense",
    "adv_suc_o": "natural adverserial accuracy under Outlier defense",
    "adv_acc_o": "natural adverserial sucess rate under Outlier defense",
    "b_adv_suc_o": "baseline  adverserial accuracy under Outlier defense",
    "b_adv_acc_o": "baseline  adverserial sucess rate under Outlier defense"}
norms = {
"natural_L_2_norm_orig":natural_L_2_norm_orig,
"natural_L_2_norm_adv":natural_L_2_norm_adv,
"natural_L_2_norm_nat":natural_L_2_norm_nat,
"natural_L_infty_norm_orig":natural_L_infty_norm_orig,
"natural_L_infty_norm_adv":natural_L_infty_norm_adv,
"natural_L_infty_norm_nat":natural_L_infty_norm_nat,
"L_2_norm_adv":L_2_norm_adv,
"L_2_norm_nat":L_2_norm_nat,
"L_infty_norm_adv":L_infty_norm_adv,
"L_infty_norm_nat":L_infty_norm_nat,
"L_cham_norm_adv":L_cham_norm_adv,
"L_cham_norm_nat":L_cham_norm_nat,
"L_emd_norm_adv":L_emd_norm_adv,
"L_emd_norm_nat":L_emd_norm_nat,
"natural_L_cham_norm_orig":natural_L_cham_norm_orig,
"natural_L_cham_norm_adv":natural_L_cham_norm_adv,
"natural_L_cham_norm_nat":natural_L_cham_norm_nat
}









L_2_norm_adv = []
L_infty_norm_adv = []
L_cham_norm_adv = []
L_emd_norm_adv = []
natural_L_cham_norm_orig = []
natural_L_cham_norm_adv = []
    norms = {
        "L_2_norm_adv": L_2_norm_adv,
        "L_infty_norm_adv": L_infty_norm_adv,
        "L_cham_norm_adv": L_cham_norm_adv,
        "L_emd_norm_adv": L_emd_norm_adv,
        "natural_L_cham_norm_orig": natural_L_cham_norm_orig,
        "natural_L_cham_norm_adv": natural_L_cham_norm_adv,
    }
    norms_names = [
        "L_2_norm_adv",
        "L_infty_norm_adv",
        "L_cham_norm_adv",
        "L_emd_norm_adv",
        "natural_L_cham_norm_orig",
        "natural_L_cham_norm_adv",
        ]
    return norms 


orig_acc = correct_classes_rate(current_scores["orig_acc"],target_class=current_scores["victim"])
adv_suc, adv_acc = attack_success_rate(current_scores["adv_acc"],target_class=current_scores["target"],victim_class=current_scores["victim"])
proj_acc = correct_classes_rate(current_scores["proj_acc"],target_class=current_scores["victim"])
rec_suc, rec_acc = attack_success_rate(current_scores["rec_acc"],target_class=current_scores["target"],victim_class=current_scores["victim"])
orig_acc_p = correct_classes_rate(current_scores["orig_acc_p"],target_class=current_scores["victim"])
adv_suc_p, adv_acc_p = attack_success_rate(current_scores["adv_acc_p"],target_class=current_scores["target"],victim_class=current_scores["victim"])
proj_acc_p = correct_classes_rate(current_scores["proj_acc_p"],target_class=current_scores["victim"])
rec_suc_p, rec_acc_p = attack_success_rate(current_scores["rec_acc_p"],target_class=current_scores["target"],victim_class=current_scores["victim"])
orig_acc_r = correct_classes_rate(current_scores["orig_acc_r"],target_class=current_scores["victim"])
adv_suc_r, adv_acc_r = attack_success_rate(current_scores["adv_acc_r"],target_class=current_scores["target"],victim_class=current_scores["victim"])
orig_acc_o = correct_classes_rate(current_scores["orig_acc_o"],target_class=current_scores["victim"])
adv_suc_o, adv_acc_o = attack_success_rate(current_scores["adv_acc_o"],target_class=current_scores["target"],victim_class=current_scores["victim"])

print("\nthe average natural norm in the original data: {}".format(np.mean(current_scores["natural_L_cham_norm_orig"])))
print("the average natural norm in the attack: {}".format(np.mean(current_scores["natural_L_cham_norm_adv"])))
# print("the average natural norm in the natural attack: {}".format(np.mean(current_scores["natural_L_cham_norm_nat"])))

print("\nthe L_infty norm to original in the attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_infty_norm_adv"]),np.min(current_scores["L_infty_norm_adv"]),np.max(current_scores["L_infty_norm_adv"])))
# print("the L_infty norm to original in the natural attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_infty_norm_nat"]),np.min(current_scores["L_infty_norm_nat"]),np.max(current_scores["L_infty_norm_nat"])))

print("\nthe L_2 norm to original in the attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_2_norm_adv"]),np.min(current_scores["L_2_norm_adv"]),np.max(current_scores["L_2_norm_adv"])))
# print("the L_2 norm to original in the natural attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_2_norm_nat"]),np.min(current_scores["L_2_norm_nat"]),np.max(current_scores["L_2_norm_nat"])))

print("\nthe L_Chamfer  norm to original in the attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_cham_norm_adv"]),np.min(current_scores["L_cham_norm_adv"]),np.max(current_scores["L_cham_norm_adv"])))
# print("the L_Chamfer  norm to original in the natural attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_cham_norm_nat"]),np.min(current_scores["L_cham_norm_nat"]),np.max(current_scores["L_cham_norm_nat"])))

print("\nthe L_EMD norm to original in the attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_emd_norm_adv"]),np.min(current_scores["L_emd_norm_adv"]),np.max(current_scores["L_emd_norm_adv"])))
# print("the L_EMD norm to original in the natural attack: avg :{:.4f} , min :{:.4f} ,  max :{:.4f}".format(np.mean(current_scores["L_emd_norm_nat"]),np.min(current_scores["L_emd_norm_nat"]),np.max(current_scores["L_emd_norm_nat"])))

print("\n POINTNET")
print(50*"-")
print("\n {} : {:5.2f}".format("the original accuracy", orig_acc))
print("the {} : {:5.2f}".format("natural adverserial accuracy", adv_acc))
print("the {}: {:5.2f}".format("natural adverserial sucess rate", adv_suc))
print("the {} : {:5.2f}".format("projected accuracy", proj_acc))
print("the {} : {:5.2f} ".format("reconstructed defense accuracy", rec_acc))
print("the {} : {:5.2f} ".format(
    "defended natural adverserial sucess rate", rec_suc))

print("\n POINTNET ++ ")
print(50*"-")
print("\nthe {}: {:5.2f}".format("original accuracy", orig_acc_p))
print("the {} : {:5.2f}".format("natural adverserial accuracy", adv_acc_p))
print("the {}: {:5.2f}".format("natural adverserial sucess rate", adv_suc_p))
print("the {} : {:5.2f}".format("projected accuracy", proj_acc_p))
print("the {} : {:5.2f} ".format("reconstructed defense accuracy", rec_acc_p))
print("the {} : {:5.2f} ".format(
    "defended natural adverserial sucess rate", rec_suc_p))

print("\n Random Defense at {} Percent ".format(10))
print(50*"-")
print("\nthe {} : {:5.2f}".format("original accuracy", orig_acc_r))
print("the {} : {:5.2f}".format("natural adverserial accuracy", adv_acc_r))
print("the {}: {:5.2f}".format("natural adverserial sucess rate", adv_suc_r))

print("\n Outlier Defense at alpha={}, K={}  ".format(1.1, 3))
print(50*"-")
print("\nthe {} : {:5.2f}".format("original accuracy", orig_acc_o))
print("the {} : {:5.2f}".format("natural adverserial accuracy", adv_acc_o))
print("the {}: {:5.2f}".format("natural adverserial sucess rate", adv_suc_o))


plt.figure()
plt.title("L infty")
seaborn.kdeplot(np.array(current_scores["L_infty_norm_adv"]), linewidth=2,
                label="baseline", clip=(0, 1))
seaborn.kdeplot(np.array(current_scores["L_infty_norm_nat"]),
                linewidth=2, label="ours", clip=(0, 1))
plt.figure()
plt.title("L 2")
seaborn.kdeplot(np.array(current_scores["L_2_norm_adv"]), linewidth=2,
                label="baseline", clip=(0, 2))
seaborn.kdeplot(np.array(current_scores["L_2_norm_nat"]), linewidth=2, label="ours", clip=(0, 2))
plt.figure()
plt.title("L Cham")
seaborn.kdeplot(1000*np.array(current_scores["L_cham_norm_adv"]),
                linewidth=2, label="baseline", clip=(0, 2))
seaborn.kdeplot(1000*np.array(current_scores["L_cham_norm_nat"]),
                linewidth=2, label="ours", clip=(0, 2))
plt.figure()
plt.title("L EMD")
seaborn.kdeplot(np.array(current_scores["L_emd_norm_adv"]), linewidth=2,
                label="baseline", clip=(0, 50))
seaborn.kdeplot(np.array(current_scores["L_emd_norm_nat"]), linewidth = 2,label="ours",clip=(0,50))

