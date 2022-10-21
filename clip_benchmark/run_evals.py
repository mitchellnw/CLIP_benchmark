import os
from cli import run, get_parser_args

if __name__ == '__main__':

    model_info = 'ViT-B-16,laion400m_e32'
    model_info_split = model_info.split(',')
    model, pretrained = model_info_split[0], model_info_split[1]

    datasets = ['cifar100', 'cars']

    for dataset in datasets:
        dataset_root = '/data/yfcc-tmp/data'
        args = get_parser_args()
        args.dataset_root = dataset_root
        args.dataset = dataset
        args.task = 'zeroshot_classification'
        args.pretrained = pretrained
        args.model = model
        args.output = 'output/' + f'{model}-{pretrained}.jsonl'.replace('/', '_')
        run(args)