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


