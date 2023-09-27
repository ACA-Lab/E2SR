# from sklearn import datasets
from e2sr import Video_test
# from e2sr_sintel import Video_test
from PSNR import get_PSNR
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8


DATA_DIR = "/home/yuzhongkai/E2SR/BasicSR/datasets/sub_test"
# DATA_DIR = "/home/songzhuoran/video/video-sr-acc/"
# dataset = "Vid4_loss"
# videoList = ["calendar", "city", "foliage", "walk"]

dirNameList = ["REDS_0", "REDS_1", "REDS_2", "REDS_3", 
               "Sintel_0", "Sintel_1", "Sintel_2", "Sintel_3",
               "GOPRO_0", "GOPRO_1", "GOPRO_2", "GOPRO_3",
               "Vid4_0", "Vid4_1", "Vid4_2", "Vid4_3"]
videoNameList = {"REDS": ["000", "001", "002", "003"],
                 "Sintel": ["ambush_1", "ambush_3", "market_1", "market_4"],
                 "GOPRO": ["GOPR0372_07_00", "GOPR0372_07_01", "GOPR0374_11_00", "GOPR0374_11_01"],
                 "Vid4": ["calendar", "city", "foliage", "walk"]}
# dirNameList = dirNameList[12: -1]
dirNameList = ["Vid4_3"]

t1List = [700, 900, 1000]
t2List = [6, 8, 12]

# t1List = [800]
# t2List = [12]

result_file = 'test_result/Basicvsr_GT.txt'
# result_file = 'test_result/Basicvsr_SR.txt'

f = open(result_file, 'w')
f.write('dataset\t video\t t1\t t2\t psnr_sr\t psnr\t psnr_ae\t bs_size\n')
f.close()
# dirNameList = [dirNameList[0]]
for dirName in dirNameList:
    dataset, idx = dirName.split("_")
    idx = int(idx)
    videoNames = videoNameList[dataset]
    videoName = videoNames[idx]

    GT_path = "%s/%s/GT/%s/" % (DATA_DIR, dirName, videoName)
    SR_path = "%s/%s/SR_result/%s/" % (DATA_DIR, dirName, videoName)
    psnr_sr = get_PSNR(GT_path, SR_path)

    for t1 in t1List:
        for t2 in t2List:
            print("generating res of %s, %d, %d" % (videoName, t1, t2))
            video_test = Video_test(DATA_DIR, dirName, videoName, t1, t2)
            video_test.cloud_server(isPar=True)
            video_test.device_ae()

            f = open(result_file, 'a+')
            # print("testing------------- %s, %d, %d" % (videoName, t1, t2))
            # video_test = Video_test(DATA_DIR, dirName, videoName, t1, t2)
            psnr = video_test.test_PSNR()
            psnr_ae = video_test.test_PSNR_ae()
            bs_size = video_test.test_bs_size()
            f.write('%s\t %s\t %04d\t %04d\t %.3f\t %.3f\t %.3f\t %.3f\n' % (dirName, videoName, t1, t2, psnr_sr, psnr, psnr_ae, bs_size))
            f.close()

            
# video_test = Video_test(DATA_DIR, dataset, "011", 800, 16)
# video_test.device()
# video_test.device_test()
# video_test.device_ae()
# video_test.res_gen()

# videoname = "calendar"

# # video_test = Video_test(DATA_DIR, dataset, videoname)
# video_test = Video_test(DATA_DIR, dataset, videoname, 1000, 100)
# # video_test.cloud_server(False)
# # video_test.device()
# # psnr = video_test.test_PSNR()
# # bs_size = video_test.test_bs_size()
# # print(psnr, bs_size)
# video_test.res_gen()