import os
import numpy as np

import torch
from torch.utils.data import Dataset

import pickle
import glob
from itertools import combinations
from itertools import permutations
from util.data_util import data_prepare_v101 as data_prepare


class S3DIS_base(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
    ):
        super().__init__()
        (
            self.split,
            self.voxel_size,
            self.transform,
            self.voxel_max,
            self.shuffle_index,
            self.loop,
        ) = (split, voxel_size, transform, voxel_max, shuffle_index, loop)

        self.data_root = data_root
        # Classes: {0:'ceiling', 1:'floor', 2:'wall', 3:'beam',
        #           4:'column', 5:'window', 6:'door', 7:'table',
        #           8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        self.class_count = 13
        class_names = open(
            os.path.join(
                os.path.dirname(os.path.dirname(data_root)),
                "meta",
                "s3dis_classnames.txt",
            )
        ).readlines()
        self.class2type = {
            i: name.strip() for i, name in enumerate(class_names)
        }
        print(self.class2type)
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.fold_0 = [
            "beam",
            "board",
            "bookcase",
            "ceiling",
            "chair",
            "column",
        ]
        self.fold_1 = ["door", "floor", "sofa", "table", "wall", "window"]

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in self.fold_1]
        else:
            raise NotImplementedError(
                "Unknown cvfold (%s). [Options: 0,1]" % cvfold
            )
        all_classes = [i for i in range(0, self.class_count - 1)]
        self.train_classes = [
            c for c in all_classes if c not in self.test_classes
        ]

        self.class2scans = self.get_class2scans()  # class, blockname

    def get_class2scans(self):
        """
        Build the class to scans mapping.
        """
        class2scans_file = os.path.join(
            os.path.dirname(self.data_root), "class2scans.pkl"
        )
        if os.path.exists(class2scans_file):
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = (
                0.05  # to filter out scans with only rare labelled points
            )
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k: [] for k in range(self.class_count)}

            for file in glob.glob(os.path.join(self.data_root, "*.npy")):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                classes = np.unique(labels)
                print(
                    "{0} | shape: {1} | classes: {2}".format(
                        scan_name, data.shape, list(classes)
                    )
                )
                for class_id in classes:
                    # if the number of points for the target class is too few,
                    # do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print("==== class to scans mapping is done ====")
            for class_id in range(self.class_count):
                print(
                    "\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}".format(
                        class_id,
                        min_ratio,
                        min_pts,
                        self.class2type[class_id],
                        len(class2scans[class_id]),
                    )
                )

            with open(class2scans_file, "wb") as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans


class S3DIS(S3DIS_base):
    """
    The S3DIS dataset class used for backbone pretraining.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
        )

        self.class2scans = {c: self.class2scans[c] for c in self.train_classes}

        train_block_names = []
        all_block_names = []
        for _, v in sorted(self.class2scans.items()):
            all_block_names.extend(v)
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])

        if split == "train":
            self.block_names = list(set(train_block_names))
        elif split == "val":
            self.block_names = list(
                set(all_block_names) - set(train_block_names)
            )
        else:
            raise NotImplementedError("Mode is unknown!")

        print(
            "[Pretrain Dataset] Mode: {0} | Num_blocks: {1}".format(
                split, len(self.block_names)
            )
        )

    def __getitem__(self, idx):
        item = self.block_names[idx % len(self.block_names)]

        data = np.load(os.path.join(self.data_root, item + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )

        class_dict = {c: i + 1 for i, c in enumerate(self.train_classes)}
        for i, lb in enumerate(label):
            if lb.item() in class_dict.keys():
                label[i] = class_dict[lb.item()]
            else:
                label[i] = 0

        return coord, feat, label

    def __len__(self):
        return len(self.block_names) * self.loop


class S3DIS_FS(S3DIS_base):
    """
    The S3DIS dataset class used for few-shot learning.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
        )

        self.n_way, self.k_shot, self.n_queries, self.num_episode = (
            n_way,
            k_shot,
            n_queries,
            num_episode,
        )

        if split == "train":
            self.classes = np.array(self.train_classes) # 从零开始的
        elif split == "test":
            self.classes = np.array(self.test_classes)
        else:
            raise NotImplementedError(
                "Unkown mode %s! [Options: train/test]" % split
            )

        self.train_mapping = {
            c: i + 1 for i, c in enumerate(self.train_classes)
        }
        print("Classes: {0} in {1} set".format(self.classes, split))

    def get_test_episode2(self, n_way_classes=None):
        """Generate a test episode without base lables."""
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )

        support_ptclouds, support_masks, query_ptclouds, query_labels, support_voxel_indexs, support_crop_indexs,\
             query_voxel_indexs, query_crop_indexs, support_ptclouds_cut, support_masks_cut, query_ptclouds_cut, query_labels_cut = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        black_list = (
            []
        )  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                ptcloud, label, voxel_index, crop_index, ptcloud_cut, label_cut = self.sample_test_pointcloud2(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_labels.append(label)
                query_voxel_indexs.append(voxel_index)
                query_crop_indexs.append(crop_index)
                query_ptclouds_cut.append(ptcloud_cut)
                query_labels_cut.append(label_cut)

            for scan_name in support_scannames:
                ptcloud, label, voxel_index, crop_index,ptcloud_cut, label_cut = self.sample_test_pointcloud2(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_masks.append(label)
                support_voxel_indexs.append(voxel_index)
                support_crop_indexs.append(crop_index)
                support_ptclouds_cut.append(ptcloud_cut)
                support_masks_cut.append(label_cut)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            sampled_classes,
            support_voxel_indexs,
            support_crop_indexs,
            query_voxel_indexs,
            query_crop_indexs,
            support_ptclouds_cut, support_masks_cut, query_ptclouds_cut, query_labels_cut,
        )
    

    def sample_test_pointcloud2(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord2, feat2, label2, voxel_index, crop_index = data_prepare(
            coord.copy(),
            feat.copy(),
            label.copy(),
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor(label)

        # construct test labels without base class labels
        feat = torch.cat((coord, feat), dim=1)
        if support:
            label = (label == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    label[i] = class_dict[lb.item()]
                else:
                    label[i] = 0

        feat2 = torch.cat((coord2, feat2), dim=1)
        if support:
            label2 = (label2 == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label2):
                if lb.item() in class_dict.keys():
                    label2[i] = class_dict[lb.item()]
                else:
                    label2[i] = 0


        return feat, label, torch.tensor(voxel_index), torch.tensor(crop_index), feat2, label2

    def get_test_episode(self, n_way_classes=None):
        """Generate a test episode without base lables."""
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )

        support_ptclouds, support_masks, query_ptclouds, query_labels = (
            [],
            [],
            [],
            [],
        )

        black_list = (
            []
        )  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_labels.append(label)

            for scan_name in support_scannames:
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            sampled_classes,
        )
    


    def sample_test_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        # construct test labels without base class labels
        feat = torch.cat((coord, feat), dim=1)
        if support:
            label = (label == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    label[i] = class_dict[lb.item()]
                else:
                    label[i] = 0

        return feat, label

    def __getitem__(self, idx, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(
                self.classes, self.n_way, replace=False
            )

        (
            support_ptclouds,
            support_base_masks,
            support_test_masks,
            query_ptclouds,
            query_base_labels,
            query_test_labels,
        ) = ([], [], [], [], [], [])

        black_list = (
            []
        )  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_base_labels.append(base_label)
                query_test_labels.append(test_label)

            for scan_name in support_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_base_masks.append(base_label)
                support_test_masks.append(test_label)
                
        return (
            support_ptclouds,
            support_base_masks,
            support_test_masks,
            query_ptclouds,
            query_base_labels,
            query_test_labels,
            sampled_classes,
            support_scannames,
            query_scannames,
        )

    def sample_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6] # N X 7, 0:3为坐标, 3:6为特征，6为标签(0~num_classes)
        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        feat = torch.cat((coord, feat), dim=1)
        if support:
            # Create a new label tensor for base calsses

            train_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in self.train_classes:
                    train_label[i] = self.train_mapping[value.item()]

            test_label = (label == sampled_class).int()
            return feat, train_label, test_label
        else:

            # Create a new label tensor for base calsses
            train_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in self.train_classes:
                    train_label[i] = self.train_mapping[value.item()]

            # Create a new label tensor for test
            test_mapping = {c: i + 1 for i, c in enumerate(sampled_classes)}
            test_label = torch.zeros_like(label)
            for i, value in enumerate(label):
                if value.item() in test_mapping.keys():
                    test_label[i] = test_mapping[value.item()]
                elif (
                    value.item() in self.test_classes and self.split == "train"
                ):
                    test_label[i] = 255

            return feat, train_label, test_label

    def __len__(self):
        return self.num_episode


class S3DIS_FSForVIS(S3DIS_FS):
    """
    The S3DIS dataset class used for visulizaling the few-shot results.
    """

    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
        target_class=None,
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
            num_episode,
            n_way,
            k_shot,
            n_queries,
        )

        self.target_class = target_class
        self.target_cls = self.type2class[self.target_class]
        self.combos = list(permutations(self.class2scans[self.target_cls], 2))

    def __getitem__(self, idx, n_way_classes=None):
        """Generate a test episode without base lables."""
        support_ptclouds, support_masks, query_ptclouds, query_labels = (
            [],
            [],
            [],
            [],
        )

        selected_scannames = self.combos[idx]
        query_scannames = selected_scannames[: self.n_queries]
        support_scannames = selected_scannames[self.n_queries :]

        for scan_name in query_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_cls], self.target_cls, support=False
            )
            query_ptclouds.append(ptcloud)
            query_labels.append(label)

        for scan_name in support_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_cls], self.target_cls, support=True
            )
            support_ptclouds.append(ptcloud)
            support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            np.array([self.target_cls]),
            selected_scannames,
        )

    def crop_point_cloud(self, point_cloud, feat, labels):
        # Count the number of points with label 1 in each half of x and y directions
        half_x = (point_cloud[:, 0].min() + point_cloud[:, 0].max()) / 2
        half_y = (point_cloud[:, 1].min() + point_cloud[:, 1].max()) / 2

        # Count points with label 1 in each quadrant
        q1_count = np.sum(
            (point_cloud[:, 0] <= half_x) & (labels == self.target_cls)
        )
        q2_count = np.sum(
            (point_cloud[:, 0] > half_x) & (labels == self.target_cls)
        )
        q3_count = np.sum(
            (point_cloud[:, 1] > half_y) & (labels == self.target_cls)
        )
        q4_count = np.sum(
            (point_cloud[:, 1] <= half_y) & (labels == self.target_cls)
        )

        # Choose the quadrant with the most points of label 1
        max_count = max(q1_count, q2_count, q3_count, q4_count)
        if max_count == q1_count:
            return (
                point_cloud[(point_cloud[:, 0] <= half_x)],
                feat[(point_cloud[:, 0] <= half_x)],
                labels[(point_cloud[:, 0] <= half_x)],
            )
        elif max_count == q2_count:
            return (
                point_cloud[(point_cloud[:, 0] > half_x)],
                feat[(point_cloud[:, 0] > half_x)],
                labels[(point_cloud[:, 0] > half_x)],
            )
        elif max_count == q3_count:
            return (
                point_cloud[(point_cloud[:, 1] > half_y)],
                feat[(point_cloud[:, 1] > half_y)],
                labels[(point_cloud[:, 1] > half_y)],
            )
        else:
            return (
                point_cloud[(point_cloud[:, 1] <= half_y)],
                feat[(point_cloud[:, 1] <= half_y)],
                labels[(point_cloud[:, 1] <= half_y)],
            )

    def sample_test_pointcloud(
        self, scan_name, sampled_classes, sampled_class, support
    ):
        data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]

        # do some crops to avoid OOM
        if support:
            while coord.shape[0] > 300000:
                print("Crop point cloud:", coord.shape[0])
                coord, feat, label = self.crop_point_cloud(coord, feat, label)
        else:
            while coord.shape[0] > 700000:
                print("Crop point cloud:", coord.shape[0])
                coord, feat, label = self.crop_point_cloud(coord, feat, label)

        coord, feat, label = data_prepare(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            # self.voxel_max,
            None,
            self.transform,
            self.shuffle_index,
            sampled_class,
        )

        # construct test labels without base class labels
        feat = torch.cat((coord, feat), dim=1)
        if support:
            label = (label == sampled_class).int()
        else:
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    label[i] = class_dict[lb.item()]
                else:
                    label[i] = 0

        return feat, label

    def __len__(self):
        return len(self.combos)


class S3DIS_FS_TEST(Dataset):
    def __init__(
        self,
        split="val",
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=50000,
        n_way=1,
        k_shot=2,
        n_queries=1,
        num_episode_per_comb=100,
    ):
        super().__init__()

        self.dataset = S3DIS_FS(
            "test",
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
            num_episode,
            n_way,
            k_shot,
            n_queries,
        )
        self.classes = self.dataset.classes
        self.n_way = n_way
        self.num_episode_per_comb = num_episode_per_comb

        if split == "val":
            self.test_data_path = os.path.join(
                os.path.dirname(data_root),
                "S_%d_N_%d_K_%d_episodes_%d_pts_%d_vs_%.2f_rdmsp"
                % (
                    cvfold,
                    n_way,
                    k_shot,
                    num_episode_per_comb,
                    voxel_max,
                    voxel_size,
                ),
            )
        elif split == "test":
            self.test_data_path = os.path.join(
                os.path.dirname(data_root),
                "S_%d_N_%d_K_%d_test_episodes_%d_pts_%d_vs_%.2f"
                % (
                    cvfold,
                    n_way,
                    k_shot,
                    num_episode_per_comb,
                    voxel_max,
                    voxel_size,
                ),
            )
        else:
            raise NotImplementedError("Mode (%s) is unknown!" % split)

    def prepare_test_data(self):
        if os.path.exists(self.test_data_path):
            self.file_names = glob.glob(
                os.path.join(self.test_data_path,  "*.pt")
            )
            self.num_episode = len(self.file_names)
        else:
            print(
                f"Test dataset ({self.test_data_path}) does not exist...\n Constructing..."
            )
            #os.mkdir(self.test_data_path)

            class_comb = list(
                combinations(self.classes, self.n_way)
            )  # [(),...]
            self.num_episode = len(class_comb) * self.num_episode_per_comb

            episode_ind = 0
            self.file_names = []
            for sampled_classes in class_comb:
                sampled_classes = list(sampled_classes)
                for _ in range(self.num_episode_per_comb):
                    data = self.dataset.get_test_episode(sampled_classes)
                    out_filename = os.path.join(
                        self.test_data_path,  f"{episode_ind}.pt"
                    )
                    if os.path.exists(out_filename):
                        break
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    episode_ind += 1

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        content =  read_episode(file_name)
        #proposals = read_episode_proposals(os.path.join(self.test_data_path, os.path.basename(file_name)))
        return content


def write_episode(out_filename, data):
    support_feat, support_label, query_feat, query_label, sampled_classes = (
        data
    )
    torch.save(
        {
            "support_feat": support_feat,
            "support_label": support_label,
            "query_feat": query_feat,
            "query_label": query_label,
            "sampled_classes": sampled_classes,
        },
        out_filename,
    )

    print("\t {0} saved! | classes: {1}".format(out_filename, sampled_classes))


def write_episode2(out_filename, data):
    support_feat, support_label, query_feat, query_label, sampled_classes, support_voxel_indexs, support_crop_indexs,\
    query_voxel_indexs, query_crop_indexs, support_ptclouds_cut, support_masks_cut, query_ptclouds_cut, query_labels_cut, = (
        data
    )
    torch.save(
        {
            "support_feat": support_feat,
            "support_label": support_label,
            "query_feat": query_feat,
            "query_label": query_label,
            "sampled_classes": sampled_classes,
            "support_voxel_indexs": support_voxel_indexs,
            "support_crop_indexs": support_crop_indexs,
            "query_voxel_indexs": query_voxel_indexs,
            "query_crop_indexs": query_crop_indexs,
            "support_ptclouds_cut": support_ptclouds_cut,
            "support_masks_cut": support_masks_cut,
            "query_ptclouds_cut": query_ptclouds_cut,
            "query_labels_cut": query_labels_cut,
        },
        out_filename,
    )

    print("\t {0} saved! | classes: {1}".format(out_filename, sampled_classes))


def read_episode(file_name):
    data_file = torch.load(file_name)
    support_feat = data_file["support_feat"]
    support_label = data_file["support_label"]
    query_feat = data_file["query_feat"]
    query_label = data_file["query_label"]
    sampled_classes = data_file["sampled_classes"]
    return (
        support_feat,
        support_label,
        query_feat,
        query_label,
        sampled_classes,
    )


def read_episode2(file_name):
    data_file = torch.load(file_name)
    support_feat = data_file["support_ptclouds_cut"]
    support_label = data_file["support_masks_cut"]
    query_feat = data_file["query_ptclouds_cut"]
    query_label = data_file["query_labels_cut"]
    sampled_classes = data_file["sampled_classes"]
    return (
        support_feat,
        support_label,
        query_feat,
        query_label,
        sampled_classes,
    )

def read_episode_proposals(file_name):
    dir_name = os.path.dirname(file_name)
    ck_name = os.path.basename(file_name)
    x = ck_name.split(".")
    file = x[0] + "_2048." + x[1]
    path = os.path.join(dir_name, 'proposals_cut_seed1024_iou_0.8_nms_0.3', ck_name)
    path_n = os.path.join(dir_name, 'proposals_cut_seed1024_iou_0.8_nms_0.3', file)
    # if os.path.exists(path_n):
    #     path = path_n
    data = torch.load(path)
    return data
