# from sklearn import datasets
from e2sr import Video_test
# from e2sr_sintel import Video_test
from PSNR import get_PSNR
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8


# DATA_DIR = "/home/yuzhongkai/E2SR/datasets/"
DATA_DIR = "/home/songzhuoran/video/video-sr-acc/"
# dataset = "Vid4_loss"
# videoList = ["calendar", "city", "foliage", "walk"]

dataVideoDic = \
    {'REDS': ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011',
              '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023',
              '024', '025', '026', '027', '028', '029'],
    'Vid4': ['calendar', 'city', 'foliage', 'walk'],
    'GOPRO': ['GOPR0372_07_00', 'GOPR0372_07_01', 'GOPR0374_11_00',  'GOPR0374_11_01', 'GOPR0374_11_02',
              'GOPR0374_11_03', 'GOPR0378_13_00', 'GOPR0379_11_00', 'GOPR0380_11_00', 'GOPR0384_11_01',
              'GOPR0384_11_02', 'GOPR0384_11_03', 'GOPR0384_11_04', 'GOPR0385_11_00', 'GOPR0386_11_00',
              'GOPR0477_11_00', 'GOPR0857_11_00', 'GOPR0868_11_01', 'GOPR0868_11_02', 'GOPR0871_11_01',
              'GOPR0881_11_00', 'GOPR0884_11_00'],
    'Sintel': ['PERTURBED_market_3', 'PERTURBED_shaman_1', 'ambush_1', 'ambush_3', 'bamboo_3', 'cave_3',
              'market_1', 'market_4', 'mountain_2', 'temple_1', 'tiger', 'wall']}


# dataset = "REDS_loss"
# dataset = "REDS"
# dataset = "Vid4"
# dataset = "Sintel"
dataset = "GOPRO"
videoList = dataVideoDic[dataset]
# videoList = videoList[0:10]
videoList = ['GOPR0374_11_01']
print(videoList)

# videoList = videoList[14:]
# videoList.remove("011")
# videoList.remove("015")
# videoList.remove("017")

t1List = [600, 700, 800]
t2List = [4, 8, 12]
# t1List = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
# t2List = [8]

result_file = 'test_result/EDVR_GT.txt'
# result_file = 'test_result/EDVR_SR.txt'
f = open(result_file, 'w')
f.write('dataset\t video\t t1\t t2\t psnr_sr\t psnr\t psnr_ae\t bs_size\n')
f.close()

for videoName in videoList:
    GT_path = "%s/%s/GT/%s/" % (DATA_DIR, dataset, videoName)
    SR_path = "%s/%s/SR_result/%s/" % (DATA_DIR, dataset, videoName)
    psnr_sr = get_PSNR(GT_path, SR_path)

    for t1 in t1List:
        for t2 in t2List:
            print("generating res of %s, %d, %d" %(videoName, t1, t2))
            video_test = Video_test(DATA_DIR, dataset, videoName, t1, t2)
            video_test.cloud_server(isPar=True)
            video_test.device_ae()
            # video_test.res_dataset_gen()
            # video_test.res_gen()
            # video_test.test_bs_size()
            # video_test.device_test()

            f = open(result_file, 'a+')
            print("testing------------- %s, %d, %d" %(videoName, t1, t2))
            psnr = video_test.test_PSNR()
            psnr_ae = video_test.test_PSNR_ae()
            bs_size = video_test.test_bs_size()
            f.write('%s %s %04d %04d %.3f %.3f %.3f %.3f\n' % (dataset, videoName, t1, t2, psnr_sr, psnr, psnr_ae, bs_size))
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