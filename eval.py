import argparse
import torch
import os
from termcolor import colored
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from collections import defaultdict
import clip
from torch.nn import functional as F

from utils.configuration import setup_config, seed_everything
from utils.fileios import *
from utils.metrics import clustering_acc
from utils.semantic_similarity import compute_semantic_similarity


from data import DATA_STATS, DATA_DISCOVERY, DATA_GROUPING, DATA_TRANSFORM
from data.utils import tta, plain_augmentation

from models.clip import build_clip
from sklearn.cluster import KMeans, DBSCAN


## vizualisation
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import VIZ.CLIP.clip as clip_viz  # uncomment for vizualisation if needed
# os.chdir(f'./Transformer-MM-Explainability')
# os.chdir(f'./VIZ')


## dev zone
print_classnames = False
print_stats = False
calculate_ub_results = False
###

def wrap_names(cname_list: list):
    try_template = "A photo of a {}, which is a bird."
    new_list = [try_template.format(cname) for cname in cname_list]
    return new_list


def unify_cnames(name: str):
    # new = name.replace('-', ' ')
    # new = new.replace("'s", '')
    new = name.strip()
    # new = new.title()
    return new


def generate_cname_classifier(cfg, encoder_t, gt_cnames: list):
    tokenized_gt_cnames = clip.tokenize(gt_cnames).to(cfg['device'])
    gt_cnames_encoding = F.normalize(encoder_t.encode_text(tokenized_gt_cnames))
    gt_cnames_list = deepcopy(gt_cnames)
    return gt_cnames_encoding, gt_cnames_list, len(gt_cnames_encoding)


def get_tta_exemplars(cfg, examplar, N_tta=10):
    T_plain = plain_augmentation(cfg['image_size'])
    T = tta(cfg['image_size'])

    og_img = T_plain(examplar).unsqueeze(0)
    og_img = og_img.to(cfg['device'])
    anchor_pool = [og_img]

    for i in range(N_tta):
        anchor = T(examplar).unsqueeze(0)
        anchor = anchor.to(cfg['device'])
        anchor_pool.append(anchor)
    return anchor_pool


def generate_voted_classifier(
        cfg,
        encoder,
        guessed_cnames: list,
        modality: str = 'single',
        alpha: float = 0.5,
        N_tta = 0,
        expt_id_suffix = ''
):
    
    ## configurable options
    upper_bound = False  # to use ground truth classnames as guessed names
    remove_duplicates = False  # to change capital letters in class-names and remove duplicates
    multi_choice = False  # to also keep variations of original class-names
    if multi_choice:
        k = 2
        use_top1_img_only = True  # to use image emb for only top1 found classname

    use_context_for_filtration = True  # use generated incotext sentences for voting
    use_context_for_classification = True  # use generated incotext sentences for final classifier
    ##


    if use_context_for_filtration or use_context_for_classification:
        if upper_bound:
            context_path = f"./data/incontext_sentences/{cfg['dataset_name']}s/{cfg['dataset_name']}s_gt_100.json" # !!! change the path !!!
        else:
            context_path = f"./data/incontext_sentences/{cfg['dataset_name']}s/{cfg['dataset_name']}s_guessed_100.json"# !!! change the path !!!

    gt_classnames = DATA_STATS[cfg['dataset_name']]['class_names']

    if upper_bound: 
        guessed_cnames = gt_classnames

    print(f"Number of guessed_cnames = {len(guessed_cnames)}")
    if remove_duplicates:
        # create list of capitatlized names:
        guessed_cnames = [x.title() for x in guessed_cnames] # For ours [] to remove duplicates. For vanilla seems like it may be already done somwhere else further
        guessed_cnames = list(set(guessed_cnames))
    print("Number of guessed_cnames without duplicates: ", len(guessed_cnames))

    guessed_cnames_lower = [x.title() for x in guessed_cnames]
    print("with duplicates: ", len(guessed_cnames_lower))
    guessed_cnames_lower = list(set(guessed_cnames_lower))
    print("without duplicates: ", len(guessed_cnames_lower))


    tfms = DATA_TRANSFORM[cfg['dataset_name']](224)
    data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)

    if len(cfg['device_ids']) > 1:
        voting_encoder, _ = build_clip('ViT-L/14', cfg['device'], jit=False, parallel=True)
        # voting_encoder, _ = build_clip('ViT-B/16', cfg['device'], jit=False, parallel=True)
    else:
        voting_encoder, _ = build_clip('ViT-L/14', cfg['device'], jit=False, parallel=False)
        # voting_encoder, _ = build_clip('ViT-B/16', cfg['device'], jit=False, parallel=False)


    if use_context_for_filtration:
        # prepare textual part with our incontext sentences for CLIP:
        photo_of = False  # use 'a photo of <class>' template
        
        path = context_path
        with open(path, 'r') as f:
            templates = json.load(f)
        
        if remove_duplicates:
            key_map_dict = {}
            for key in templates.keys():
                key_map_dict[key] = key.title()
            templates = {(key_map_dict[k] if k in key_map_dict else k):v  for (k,v) in templates.items() }

        basic_templates = ['a photo of a {}.']
            
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(guessed_cnames):  
                try:
                    texts = templates[classname] # format with class
                    if photo_of:
                        template_photo = basic_templates
                        texts = [template.format(text) for text in texts for template in template_photo] #format with class
                    else:
                        texts = texts

                except Exception as e:
                    print(f"[WARNING] {e}")
                    print("[WARNING] No incontext sentences found for classname:", classname, ", use class name as a context instead.")

                    texts = classname.split(",")[0] if classname else [classname]
                    if photo_of:
                        template_photo = basic_templates
                    texts = [template.format(classname) for template in templates] #format with class

                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = voting_encoder.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        vote_cnames_encoding = zeroshot_weights.T
    else:
        vote_tokenized_cnames = clip.tokenize(guessed_cnames).to(cfg['device'])
        vote_cnames_encoding = F.normalize(voting_encoder.encode_text(vote_tokenized_cnames))


    if modality == 'single':
        # voting
        candidates_indices = []
        for idx, (img, label) in tqdm(enumerate(data_discovery)):
            img = tfms(img).unsqueeze(0)
            img = img.to(cfg['device'])

            img_encoding = voting_encoder.encode_image(img)
            img_encoding = F.normalize(img_encoding)

            score_clip = img_encoding @ vote_cnames_encoding.T
            idx_top1 = score_clip.argmax(dim=1)
            candidates_indices.append(int(idx_top1[0]))

        # choose the candidates after voting
        candidates_indices = list(set(candidates_indices))
        print(f"Number of selected candidates = {len(candidates_indices)}")
        
        print("Candidate names:")
        candidates_name = [guessed_cnames_lower[i] for i in candidates_indices]
        print(len(candidates_name))
        print(candidates_name)
        candidates_name = [x.title() for x in candidates_name]
        candidates_name = list(set(candidates_name))
        print(len(candidates_name))

        print("Removed classnames:")
        removed_cnames = set(guessed_cnames_lower) - set(candidates_name)
        print(len(removed_cnames))
        print(removed_cnames)
        removed_cnames = [x.title() for x in removed_cnames]
        removed_cnames = list(set(removed_cnames))
        print(len(removed_cnames))

    elif modality == 'cross':
        if print_stats: 
            images_per_class_stats = defaultdict(int)

        candidates_pairs = defaultdict(list)

        ''' get stats (uncomment if needed)
        top_5_similarity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        top_5_names = [None, None, None, None, None, None, None, None, None, None]
        top_5_img_paths = [None, None, None, None, None, None, None, None, None, None]

        bottom_5_similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        bottom_5_names = [None, None, None, None, None, None, None, None, None, None]
        bottom_5_img_paths = [None, None, None, None, None, None, None, None, None, None]
        '''

        for idx, (img, label, img_path) in tqdm(enumerate(data_discovery)):
            img_vote = tfms(img).unsqueeze(0)
            img_vote = img_vote.to(cfg['device'])

            img_vote_encoding = voting_encoder.encode_image(img_vote)
            img_vote_encoding = F.normalize(img_vote_encoding)

            score_clip = img_vote_encoding @ vote_cnames_encoding.T

            ''' get stats (uncomment if needed)
            prediction_top_5 = score_clip.topk(1, dim=1).indices
            prediction_top_5_similarity = score_clip.topk(1, dim=1).values
            prediction_top_5_similarity = prediction_top_5_similarity.cpu().numpy()
            prediction_top_5_similarity = prediction_top_5_similarity.tolist()
            print("current batch top-5 similarity for 1 image:", prediction_top_5_similarity[0])
            print("current batch top-5 names for 1 image:", [guessed_cnames[pred_idx] for pred_idx in prediction_top_5[0]])        

            # update global top-5 if needed new top results are found
            for batch_id in range(len(prediction_top_5)):
                for i in range(len(prediction_top_5[batch_id])):
                    if prediction_top_5_similarity[batch_id][i] > min(top_5_similarity):
                        min_idx = top_5_similarity.index(min(top_5_similarity))
                        top_5_similarity[min_idx] = prediction_top_5_similarity[batch_id][i]
                        # print(prediction_top_5.shape)
                        # print(prediction_top_5[batch_id].shape)
                        # print(prediction_top_5[batch_id][i].shape)
                        top_5_names[min_idx] = guessed_cnames[prediction_top_5[batch_id][i]]
                        top_5_img_paths[min_idx] = img_path
            print("global top-5 similarity:", top_5_similarity)
            print("global top-5 names:", top_5_names)
            print("global top-5 image paths:", top_5_img_paths)


            prediction_bottom_5 = score_clip.topk(1, dim=1).indices  #, largest=False).indices
            prediction_bottom_5_similarity = score_clip.topk(1, dim=1).values  #, largest=False).values
            prediction_bottom_5_similarity = prediction_bottom_5_similarity.cpu().numpy()
            prediction_bottom_5_similarity = prediction_bottom_5_similarity.tolist()
            print("current batch bottom-5 similarity for batch:", prediction_bottom_5_similarity[0])
            print("current batch bottom-5 names for 1 image:", [guessed_cnames[pred_idx] for pred_idx in prediction_bottom_5[0]])

            # update global bottom-5 if needed new bottom results are found
            for batch_id in range(len(prediction_bottom_5)):
                for i in range(len(prediction_bottom_5[batch_id])):
                    if prediction_bottom_5_similarity[batch_id][i] < max(bottom_5_similarity):
                        max_idx = bottom_5_similarity.index(max(bottom_5_similarity))
                        bottom_5_similarity[max_idx] = prediction_bottom_5_similarity[batch_id][i]
                        bottom_5_names[max_idx] = guessed_cnames[prediction_bottom_5[batch_id][i]]
                        bottom_5_img_paths[max_idx] = img_path
            print("global bottom-5 similarity:", bottom_5_similarity)
            print("global bottom-5 names:", bottom_5_names)
            print("global bottom-5 image paths:", bottom_5_img_paths)
            '''

            if not multi_choice:
                idx_top1 = score_clip.argmax(dim=1)
                idx_top1 = int(idx_top1[0])
                # print("top-1: ", idx_top1)
            else:
                val_topK, idx_topK = torch.topk(score_clip, k, dim=1)
                # print("top-K raw: ", idx_topK)
                # print("top-K raw values: ", val_topK)
                idx_top1 = int(idx_topK[0][0])
                idx_topK = [int(idx) 
                                for id_temp, idx in enumerate(idx_topK[0])
                                    if (id_temp == 0) or ( val_topK[0][id_temp] > (0.8 * val_topK[0][0]) )
                            ]
                
                if len(idx_topK) < k:
                    print("Ignoring lower top-K due to low confidence")
                # print("top-K: ", idx_topK)
            
            if not multi_choice:
                candidates_pairs[idx_top1].extend(
                    get_tta_exemplars(cfg, img, N_tta=N_tta)
                )
            else:
                for idx_top in idx_topK:
                    if (idx_top == idx_top1) or (not use_top1_img_only): # top-1
                        candidates_pairs[idx_top].extend(
                            get_tta_exemplars(cfg, img, N_tta=N_tta)
                        )
                    else: # top-K
                        if idx_top not in candidates_pairs:
                            candidates_pairs[idx_top] = []
                
            if print_stats: 
                images_per_class_stats[guessed_cnames[idx_top1]] += 1

        if print_stats: 
            # print stats sorted by key:
            images_per_class_stats = dict(sorted(images_per_class_stats.items(), key=lambda item: item[0]))
            # # print stats sorted by value:
            # images_per_class_stats = dict(sorted(images_per_class_stats.items(), key=lambda item: item[1], reverse=True))

            print(f"Stats filtration before num = {len(set(list(images_per_class_stats.keys())))}")
            print(f"Stats filtration before: {images_per_class_stats}")

        ''' get stats (uncomment if needed)
        # keys_to_remove = []
        classnames_to_remove = []
        for key, value in candidates_pairs.items():
            # if len(value) == 0:
            #     keys_to_remove.append(key)
            elif len(value) <= (N_tta+1):
                print(f"Class {guessed_cnames[key]} has only {len(value)} exemplars, not using image emb for it") #removing it from candidates")
                classnames_to_remove.append(guessed_cnames[key])
                candidates_pairs[key] = []
        # candidates_pairs = {k: v for k, v in candidates_pairs.items() if k not in keys_to_remove}

        images_per_class_stats = {k: v 
                                    for k, v in images_per_class_stats.items() 
                                        if k not in classnames_to_remove}
        print(f"Stats filtration after num = {len(set(list(images_per_class_stats.keys())))}")
        print(f"Stats filtration after: {images_per_class_stats}")
        '''

        print(f"Number of selected candidates = {len(set(list(candidates_pairs.keys())))}")
        print("Candidate names:")
        candidates_name = [guessed_cnames[i] for i in list(set(list(candidates_pairs.keys())))]
        print(len(candidates_name))
        if print_classnames: print(candidates_name)
        candidates_name = [x.title() for x in candidates_name]
        candidates_name = list(set(candidates_name))
        print(len(candidates_name))

        print("Removed classnames:")
        removed_cnames = set(guessed_cnames_lower) - set(candidates_name)
        print(len(removed_cnames))
        if print_classnames: print(removed_cnames)
        removed_cnames = [x.title() for x in removed_cnames]
        removed_cnames = list(set(removed_cnames))
        print(len(removed_cnames))

    else:
        raise NotImplementedError

    del voting_encoder

    print("GT names:")
    print(len(gt_classnames))
    if print_classnames: print(gt_classnames)
    gt_classnames = [x.title() for x in gt_classnames]
    gt_classnames = list(set(gt_classnames))
    print(len(gt_classnames))

    print("GT excluded names CLIP:")
    gt_excluded_names = set(gt_classnames).intersection(set(removed_cnames))
    print(len(gt_excluded_names))
    if print_classnames: print(gt_excluded_names)

    print("GT included names CLIP:")
    gt_included_names = set(gt_classnames).intersection(set(candidates_name))
    print(len(gt_included_names))
    if print_classnames: print(gt_included_names)

    texts_viz = None
    if use_context_for_classification:
        # prepare textual part with our incontext sentences for CLIP:
        photo_of = False  # use 'a photo of <class>' template

        path = context_path
        with open(path, 'r') as f:
            templates = json.load(f)
        
        if remove_duplicates:
            key_map_dict = {}
            for key in templates.keys():
                key_map_dict[key] = key.title()

            templates = {(key_map_dict[k] if k in key_map_dict else k):v  for (k,v) in templates.items() }

        basic_templates = ['a photo of a {}.']

        texts_viz = {}
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(guessed_cnames):
                try:
                    texts = templates[classname] # format with class
                    if photo_of:
                        template_photo = basic_templates
                        texts = [template.format(text) for text in texts for template in template_photo] #format with class
                    else:
                        texts = texts
                except Exception as e:
                    print(f"[WARNING] {e}")
                    print("[WARNING] No incontext sentences found for classname:", classname, ", use class name as a context instead.")

                    texts = classname.split(",")[0] if classname else [classname]
                    if photo_of:
                        template_photo = basic_templates
                    texts = [template.format(classname) for template in templates] #format with class

                texts_viz[classname] = texts

                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = encoder.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            vote_cnames_encoding = zeroshot_weights.T
        cnames_encoding = vote_cnames_encoding

    else:
        tokenized_cnames = clip.tokenize(guessed_cnames).to(cfg['device'])
        cnames_encoding = F.normalize(encoder.encode_text(tokenized_cnames))

    # build final classifier
    if modality == 'single':
        selected_classifier = cnames_encoding[candidates_indices]
        selected_names = [guessed_cnames[i] for i in candidates_indices]
    elif modality == 'cross':
        selected_classifier = []
        for k, v in candidates_pairs.items():
            vec_txt = cnames_encoding[k]

            if multi_choice and (len(v) == 0):
                vec_mixed = vec_txt
            else:
                v = torch.concat(v, dim=0)
                vec_img = encoder.encode_image(v)
                vec_img = F.normalize(vec_img)
                vec_img = vec_img.mean(dim=0)

                vec_mixed = alpha * vec_txt + (1 - alpha) * vec_img

            selected_classifier.append(vec_mixed.view(1, -1))
        selected_classifier = torch.concat(selected_classifier, dim=0)
        selected_names = [guessed_cnames[k] for k, _ in candidates_pairs.items()]
    else:
        raise NotImplementedError

    if texts_viz is None:
        texts_viz = selected_names

    return selected_classifier, selected_names, len(selected_classifier), texts_viz


def main_eval(cfg, data_grouping, gt_category_sheet, encoder, classifier, cls_name_list, gt=False, texts_viz=None, gt_name_list_viz=None):

    if gt:
        print("---> Evaluating w/ GT")
        suffix = "gt"
    else:
        print("---> Evaluating w/ ours")
        suffix = "ours"

    total_preds= np.array([])
    total_labels = np.array([])

    total_pred_names = []
    total_label_names = []
    total_img_paths = []    # for visualization

    ''' get stats (uncomment if needed)
    top_5_similarity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    top_5_names = [None, None, None, None, None, None, None, None, None, None]
    top_5_img_paths = [None, None, None, None, None, None, None, None, None, None]

    bottom_5_similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bottom_5_names = [None, None, None, None, None, None, None, None, None, None]
    bottom_5_img_paths = [None, None, None, None, None, None, None, None, None, None]
    '''

    if print_stats: 
        images_per_class_stats_final = defaultdict(int)

    do_vizuals = False  # set to True to enable vizualizations
    method = "ours" # [ ours, finer ]
    if do_vizuals:      
        separate_sentences = True  # to process each sentence separately
        model, preprocess = clip_viz.load("ViT-B/16", device=cfg['device'], jit=False)
        os.makedirs(f"./vizuals/{cfg['dataset_name']}", exist_ok=True)

    correct_num = 0
    for batch_idx, (images, labels, img_paths) in enumerate(tqdm(data_grouping)):

        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)

        similarity = image_encodings @ classifier.T
        # similarity = F.softmax(similarity/0.1, dim=1)
        #print(similarity)
        prediction = similarity.argmax(dim=1)

        ''' get stats (uncomment if needed)
        prediction_top_5 = similarity.topk(1, dim=1).indices
        prediction_top_5_similarity = similarity.topk(1, dim=1).values
        prediction_top_5_similarity = prediction_top_5_similarity.cpu().numpy()
        prediction_top_5_similarity = prediction_top_5_similarity.tolist()
        print("current batch top-5 similarity for 1 image:", prediction_top_5_similarity[0])
        print("current batch top-5 names for 1 image:", [cls_name_list[pred_idx] for pred_idx in prediction_top_5[0]])        

        # update global top-5 if needed new top results are found
        for batch_id in range(len(prediction_top_5)):
            for i in range(len(prediction_top_5[batch_id])):
                if prediction_top_5_similarity[batch_id][i] > min(top_5_similarity):
                    min_idx = top_5_similarity.index(min(top_5_similarity))
                    top_5_similarity[min_idx] = prediction_top_5_similarity[batch_id][i]
                    #print(prediction_top_5.shape)
                    #print(prediction_top_5[batch_id].shape)
                    #print(prediction_top_5[batch_id][i].shape)
                    top_5_names[min_idx] = cls_name_list[prediction_top_5[batch_id][i]]
                    top_5_img_paths[min_idx] = img_paths[batch_id]
        print("global top-5 similarity:", top_5_similarity)
        print("global top-5 names:", top_5_names)
        print("global top-5 image paths:", top_5_img_paths)


        prediction_bottom_5 = similarity.topk(1, dim=1, largest=False).indices
        prediction_bottom_5_similarity = similarity.topk(1, dim=1, largest=False).values
        prediction_bottom_5_similarity = prediction_bottom_5_similarity.cpu().numpy()
        prediction_bottom_5_similarity = prediction_bottom_5_similarity.tolist()
        print("current batch bottom-5 similarity for 1 image:", prediction_bottom_5_similarity[0])
        print("current batch bottom-5 names for 1 image:", [cls_name_list[pred_idx] for pred_idx in prediction_bottom_5[0]])

        # update global bottom-5 if needed new bottom results are found
        for batch_id in range(len(prediction_bottom_5)):
            for i in range(len(prediction_bottom_5[batch_id])):
                if prediction_bottom_5_similarity[batch_id][i] < max(bottom_5_similarity):
                    max_idx = bottom_5_similarity.index(max(bottom_5_similarity))
                    bottom_5_similarity[max_idx] = prediction_bottom_5_similarity[batch_id][i]
                    bottom_5_names[max_idx] = cls_name_list[prediction_bottom_5[batch_id][i]]
                    bottom_5_img_paths[max_idx] = img_paths[batch_id]
        print("global bottom-5 similarity:", bottom_5_similarity)
        print("global bottom-5 names:", bottom_5_names)
        print("global bottom-5 image paths:", bottom_5_img_paths)
        '''

        names_prediction = [cls_name_list[pred_idx] for pred_idx in prediction]

        if print_stats: 
            for pred_idx in prediction:
                images_per_class_stats_final[cls_name_list[pred_idx]] += 1

        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction.cpu().numpy())
        total_pred_names.extend(names_prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())
        names_label = [gt_category_sheet[gt_idx] for gt_idx in labels]
        total_label_names.extend(names_label)
        #       |- image path
        total_img_paths.extend(img_paths)


        if do_vizuals and (correct_num <= 30): # (batch_idx <= 3 or correct_num <= 10):

            if gt_name_list_viz is None:
                raise ValueError("gt_name_list_viz is not None, please provide it")

            for image_id, image in enumerate(images):

                if cls_name_list[prediction[image_id]] != gt_name_list_viz[labels.cpu().numpy()[image_id]]:
                    suffix_pred = "wrong"
                    # print(f"GT: {gt_name_list_viz[labels.cpu().numpy()[image_id]]}, {labels.cpu().numpy()[image_id]}")
                    # print(f"pred: {cls_name_list[prediction[image_id]]}, {prediction[image_id]}")
                    # continue
                else:
                    suffix_pred = "correct"
                    correct_num += 1

                print(f"GT: {gt_name_list_viz[labels.cpu().numpy()[image_id]]}, {labels.cpu().numpy()[image_id]}")
                print(f"pred: {cls_name_list[prediction[image_id]]}, {prediction[image_id]}")

                img_path = img_paths[image_id]
                img_name = img_path.split("/")[-1]
                print(f"img_name: {img_name}")
                os.makedirs(f"./vizuals/{cfg['dataset_name']}/{method}/{suffix_pred}/{img_name}", exist_ok=True)

                img = preprocess(Image.open(img_path)).unsqueeze(0).to(cfg['device'])

                if texts_viz is not None:
                    if method == "ours":
                        texts_save = texts_viz[cls_name_list[prediction[image_id]]]
                    else:
                        texts_save = [cls_name_list[prediction[image_id]]]
                else:
                    raise ValueError("texts_viz is None, please provide it")

                # save texts  in a txt file:
                with open(f"./vizuals/{cfg['dataset_name']}/{method}/{suffix_pred}/{img_name}/_texts.txt", "w") as f:
                    for text_save in texts_save:
                        f.write(text_save + "\n")

                if separate_sentences:
                    # texts = ['a zebra', 'an elephant', 'a lake']
                    # texts = [cls_name_list[i] for i in range(len(text))]
                    ## texts = [cls_name_list[prediction[image_id]]]

                    if method == "ours":
                        texts = texts_viz[cls_name_list[prediction[image_id]]]
                    else:
                        texts = [cls_name_list[prediction[image_id]]]

                    text = clip_viz.tokenize(texts).to(cfg['device'])
                    # print(text.size())

                else:
                    texts = ["combined"]
                    raise NotImplementedError

                R_text, R_image = interpret(model=model, image=img, texts=text, device=cfg['device'])
                batch_size = text.shape[0]
                for i in range(batch_size):
                    if i >= 10:
                        break
                    # text_temp = texts[i]
                    # remove forward slash from the text:
                    # text_temp = text_temp.replace("/", "_")

                    # show_heatmap_on_text(texts[i], text[i], R_text[i]) # works only in notebooks (html visualization)
                    show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))
                    try:
                        plt.savefig(f"./vizuals/{cfg['dataset_name']}/{method}/{suffix_pred}/{img_name}/{i}_{texts[i]}.png") #, bbox_inches='tight')
                    except Exception as e:
                        print(f"[WARNING] {e}")
                        plt.savefig(f"./vizuals/{cfg['dataset_name']}/{method}/{suffix_pred}/{img_name}/{i}_text.png")
                    # plt.show()

                if correct_num >= 5:
                    break

    if do_vizuals:
        del model


    if print_stats: 
        # print stats sorted by key:
        images_per_class_stats_final = dict(sorted(images_per_class_stats_final.items(), key=lambda item: item[0]))
        # # print stats sorted by value:
        # images_per_class_stats_final = dict(sorted(images_per_class_stats_final.items(), key=lambda item: item[1], reverse=True))

        print(f"Stats final w/ {suffix}: {images_per_class_stats_final}")
        print(f"Stats final w/ {suffix} num = {len(set(list(images_per_class_stats_final.keys())))}")

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)

    del encoder
    # torch.cuda.empty_cache()

    tryout_sacc_model_zoo = ['sbert_base']
    for try_model in tryout_sacc_model_zoo:
        results[f'ssACC_{try_model}'] = compute_semantic_similarity(total_pred_names, total_label_names,
                                                                    model=try_model, device=cfg['device'],
                                                                    device_ids=cfg['device_ids'])
    return results


def kmeans_eval(cfg, data_grouping, encoder, cluster):
    print("---> Evaluating w/ KMeans")
    total_preds= np.array([])
    total_labels = np.array([])

    for batch_idx, (images, labels, _) in enumerate(tqdm(data_grouping)):
        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        image_encodings = image_encodings.cpu().numpy()

        prediction = cluster.predict(image_encodings)

        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)
    return results


def dbscan_eval(cfg, data_grouping, encoder, cluster):
    print("---> Evaluating w/ KMeans")
    total_preds= np.array([])
    total_labels = np.array([])

    for batch_idx, (images, labels, _) in enumerate(tqdm(data_grouping)):
        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        image_encodings = image_encodings.cpu().numpy()

        prediction = cluster.fit_predict(image_encodings)

        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)
    return results


def print_results(results: dict, method: str = 'clip'):
    method_name = method.upper()

    print("\n")
    print(colored("=" * 25 + f" {method_name}-based Final Results " + "=" * 25, "yellow"))
    print("\n")
    print(f"[Clustering]")
    print(f"Total {method_name}-based Clustering Acc: {100 * results['acc_clustering']}")
    print(f"Total {method_name}-based Clustering Nmi: {100 * results['nmi_clustering']}")
    print(f"Total {method_name}-based Clustering Ari: {100 * results['ari_clustering']}")
    print("\n")
    print(f"[ssACC (semantic similarity ACC]")
    for try_model in ['sbert_base']:
        print(f"ssACC_{try_model}: {100 * results[f'ssACC_{try_model}']}")
    print(colored("=" * 25 + "          END          " + "=" * 25, "yellow"))


## visualization and interpretation:
    
#@title Control context expansion (number of attention layers to consider)
#@title Number of layers for image Transformer
start_layer =  -1 #@param {type:"number"}

#@title Number of layers for text Transformer
start_layer_text =  -1 #@param {type:"number"}

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      # calculate index of last layer 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
      # calculate index of last layer 
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance


def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');


''' for visualizing text relevance
from VIZ.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    # print(text_scores)
    text_tokens=_tokenizer.encode(text)
    text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
    visualization.visualize_text(vis_data_records)
'''
##


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grouping', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--config_file_env',
                        type=str,
                        default='./configs/env_machine.yml',
                        help='location of host environment related config file')
    parser.add_argument('--config_file_expt',
                        type=str,
                        default='./configs/expts/bird200_all.yml',
                        help='location of host experiment related config file')
    # Hyper-parameters
    parser.add_argument('--alpha',
                        type=float,
                        default=0.7)
    parser.add_argument('--N_tta',
                        type=int,
                        default=10)
    # arguments for control experiments
    parser.add_argument('--num_per_category',
                        type=str,
                        default='3',
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'],
                        )
    parser.add_argument('--num_runs',
                        type=int,
                        default=10)


    # init. configuration
    args = parser.parse_args()
    cfg = setup_config(args.config_file_env, args.config_file_expt)
    print(colored(args, 'yellow'))

    # drop the seed
    seed_everything(cfg['seed'])

    expt_id_suffix = f"_{args.num_per_category}"

    device_count = torch.cuda.device_count()
    print("Number of GPUs:", device_count)

    for i in range(device_count):
        print("Device ID:", i, "Device Name:", torch.cuda.get_device_name(i))

    device_ids = [i for i in range(device_count)]
    cfg['device'] = "cuda"
    cfg['device_ids'] = device_ids
    # cfg['device'] = 'cpu'

    # build names
    gt_cnames = DATA_STATS[cfg['dataset_name']]['class_names']
    gt_category_sheet = deepcopy(gt_cnames)
    guessed_cnames = load_json(cfg['path_llm_gussed_names'] + expt_id_suffix)
    guessed_cnames = [unify_cnames(cname) for cname in guessed_cnames]

    print("Guessed names:")
    if print_classnames: print(guessed_cnames)

    # build VLM model
    if len(cfg['device_ids']) > 1:
        encoder, preprocesser = build_clip(cfg['model_size'], cfg['device'], jit=False, parallel=True)
    else:
        encoder, preprocesser = build_clip(cfg['model_size'], cfg['device'], jit=False, parallel=False)

    # build dataloaders
    data_grouping = DATA_GROUPING[cfg['dataset_name']](cfg)


    vilang_cACC = 0.0
    vilang_sACC = 0.0
    for i in range(args.num_runs):
        # generate classifier
        #   |- upper bound
        #if calculate_ub_results: 
        gt_classifier, gt_name_list, len_gt_classifier = generate_cname_classifier(cfg, encoder, gt_cnames)

        vilang_classifier, vilang_name_list, len_vilang_classifier, texts_viz = generate_voted_classifier(
            cfg, encoder, guessed_cnames, modality='cross', alpha=args.alpha, N_tta=args.N_tta,
            expt_id_suffix=expt_id_suffix,
        )
        #vilang_name_list = guessed_cnames  # use real text for debug and visualization purpose

        print("---> Each Classifier' shapes")
        if calculate_ub_results: print(f"\t GT_classifier = {len_gt_classifier}")
        print(f"\t ViLang_guessed = {len_vilang_classifier}")

        # run the main script
        if calculate_ub_results:
            gt_results = main_eval(cfg, data_grouping, gt_category_sheet, encoder, gt_classifier, gt_name_list, gt=True)
        vilang_results = main_eval(cfg, data_grouping, gt_category_sheet, encoder, vilang_classifier, vilang_name_list, gt=False, texts_viz=texts_viz, gt_name_list_viz=gt_name_list)

        if calculate_ub_results: print_results(gt_results, method="UpperBound: CLIP zero-shot")
        print_results(vilang_results, method=f"Ours: ViLangGuessed w/ alpha={args.alpha}, N_tta={args.N_tta}")

        vilang_cACC += vilang_results['acc_clustering']
        vilang_sACC += vilang_results['ssACC_sbert_base']

    vilang_cACC /= args.num_runs
    vilang_sACC /= args.num_runs

    print("\n")
    print(colored("=" * 25 + f" ViLang Final Results of {args.num_runs} runs, w/ {args.num_per_category} imgs per class"
                  + "=" * 25, "yellow"))
    print("\n")
    print(f"[Clustering]")
    print(f"Clustering ACC: {100*vilang_cACC}")
    print(f"Semantic ACC:   {100*vilang_sACC}")
    print(colored("=" * 25 + "          END          " + "=" * 25, "yellow"))
