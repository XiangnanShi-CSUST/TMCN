from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, MDetailFeatureExtraction, M2BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import setproctitle
import clip
import cv2

warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.CRITICAL)

output_file = "output.txt"
logging.basicConfig(filename=output_file, level=logging.INFO, format='%(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path="TMCN.pth"


print(ckpt_path)
for dataset_name in ["TNO", "RoadScene", "MSRS"]:

    print("\n"*2+"="*80)
    model_name="TMCN    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text_line = "In this challenge, we're addressing the fusion of infrared and visible images, with a specific focus on the low contrast degradation in the infrared images."
    # text_line = "In this challenge, we're addressing the fusion of infrared and visible images."
    model_clip, _ = clip.load("ViT-B/32", device=device)

    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder(model_clip = model_clip)).to(device)
    BaseFuseLayer = nn.DataParallel(M2BaseFeatureExtraction(dim=64, hidden_dim=64, model_clip = model_clip)).to(device)
    DetailFuseLayer = nn.DataParallel(MDetailFeatureExtraction(num_layers=1,  hidden_dim=64, model_clip = model_clip)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):


            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][
                           np.newaxis, np.newaxis, ...] / 255.0

            # ycrcb, uint8
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            text = clip.tokenize(text_line).to(device)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B, text)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D, text)

            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D, text)

            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8) # 改的防止浮点数

            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder  
    ori_img_folder=test_folder

    logging.info("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tAG\tCC")

    metric_result = np.zeros((10))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                        , Evaluator.AG(fi), Evaluator.CC(fi, ir, vi)])

            logging.info(img_name + '\t' + f"{Evaluator.EN(fi):.2f}" + '\t'
                         + f"{Evaluator.SD(fi):.2f}" + '\t'
                         + f"{Evaluator.SF(fi):.2f}" + '\t'
                         + f"{Evaluator.MI(fi, ir, vi):.2f}" + '\t'
                         + f"{Evaluator.SCD(fi, ir, vi):.2f}" + '\t'
                         + f"{Evaluator.VIFF(fi, ir, vi):.2f}" + '\t'
                         + f"{Evaluator.Qabf(fi, ir, vi):.2f}" + '\t'
                         + f"{Evaluator.SSIM(fi, ir, vi):.2f}" + '\t'
                         + f"{Evaluator.AG(fi):.2f}" + '\t'
                         + f"{Evaluator.CC(fi, ir, vi):.2f}"
                         )
    logging.info("=" * 100)

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tAG\tCC")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))+'\t'
            +str(np.round(metric_result[8], 2))+'\t'
            +str(np.round(metric_result[9], 2))
            )
    print("="*100)