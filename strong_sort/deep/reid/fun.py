from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import os
import os.path as osp

from torchreid.data import ImageDataset
import torchreid


class NewDataset(ImageDataset):
    # dataset_dir = 'ReID'

    def __init__(self, root='', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))  # self.root = 'E:\\a_ml_project\\pro_reid\\reid-data'
        # self.root = '/data1/wangsw/ReID'  # self.root = 'E:\\a_ml_project\\pro_reid\\reid-data'
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)  # 'E:\\a_ml_project\\pro_reid\\reid-data\\new_dataset'
        self.dataset_dir = '/data1/wangsw/ReID/bounding_box_train'

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        query = []
        gallery = []
        train = []
        imgs_list = os.listdir(self.dataset_dir)
        imgs_list.sort()

        last_id = 0
        add_id = -1
        for img_name in imgs_list:
            path = osp.join(self.dataset_dir, img_name)
            camid = 0

            now_id = int(img_name.split('_')[0])
            if now_id == last_id:
                train.append((path, add_id, camid))
            if now_id != last_id:
                add_id += 1
                train.append((path, add_id, camid))
            last_id = now_id

        query_dir = '/data1/wangsw/ReID/query'
        query_imgs_list = os.listdir(query_dir)
        query_imgs_list.sort()
        for query_img_name in query_imgs_list:
            path = osp.join(query_dir, query_img_name)
            caid = 0
            ad_id = int(query_img_name.split('_')[0])
            query.append((path,ad_id,caid))

        gallery_dir = '/data1/wangsw/ReID/gallery'
        gallery_imgs_list = os.listdir(gallery_dir)
        gallery_imgs_list.sort()
        for gallery_img_name in gallery_imgs_list:
            path = osp.join(gallery_dir, gallery_img_name)
            caid = 1
            ad_id = int(gallery_img_name.split('_')[0])
            gallery.append((path, ad_id, caid))

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)


torchreid.data.register_image_dataset('ReID', NewDataset)

datamanager = torchreid.data.ImageDataManager(
    # root='reid-data',
    sources='ReID',
    batch_size_train=64
)
model = torchreid.models.build_model(
    name="osnet_ain_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="triplet",
    pretrained=False,
    use_gpu = True
)
# weight_path = '/home/wangsw/projects/pro_visdrone_new/Yolov5_StrongSORT_OSNet/weights/reid_ped/osnet_ain_x1_0_duke2mar_cos.pth'
# torchreid.utils.load_pretrained_weights(model, weight_path)

model = model.to("cuda:0")
print("123")
optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)
#
# engine = torchreid.engine.ImageSoftmaxEngine(
#     datamanager,
#     model,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     label_smooth=True
# )
engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    weight_t=1,
    weight_x=1,

    label_smooth=True
)

engine.run(
    save_dir="log/my_reid",
    max_epoch=30,
    eval_freq=5,
    print_freq=10,
    test_only=False,
    dist_metric='cosine',

)

