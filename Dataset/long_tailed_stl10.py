import numpy as np
from Dataset.dataset import label_indices2indices
import copy


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    """
    클래스별 이미지 개수를 불균형 계수에 따라 계산
    """
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail_stl10(list_label2indices_train, num_classes, imb_factor, imb_type):
    """
    STL-10용 long-tailed 데이터셋 생성 함수
    
    Args:
        list_label2indices_train: 클래스별 인덱스 리스트 (list of lists)
        num_classes: 클래스 개수 (STL-10은 10)
        imb_factor: 불균형 계수 (예: 0.01, 0.1 등)
        imb_type: 불균형 타입 ('exp' 등)
    
    Returns:
        img_num_list: 클래스별 샘플 개수 리스트
        list_clients_indices: 클래스별 선택된 인덱스 리스트
    """
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('STL-10 img_num_class (long-tailed distribution)')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('STL-10 All num_data_train (after long-tailed sampling)')
    print(len(num_list_clients_indices))
    return img_num_list, list_clients_indices

