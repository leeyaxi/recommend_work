import argparse
import torch
import os 

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
    parser.add_argument('--test_size', type=int, default=64, help='input batch size')

    parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size')
    parser.add_argument('--layer', type=int, default=3)

    parser.add_argument('--epoch', type=int, default=1, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--topk', default=[10, 20], type=list)

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=1000, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')  
    parser.add_argument('--patience', type=float, default=5 )   
    
    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: args.device = torch.device('cpu')
    return args
