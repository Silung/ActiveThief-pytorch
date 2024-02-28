import optuna
from mea import *

def optuna_search(args):
    def objective(trial):
        # 定义需要调优的超参数空间
        args.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        args.patience = trial.suggest_int('patience', 10, 200)
        args.mea_dropout = trial.suggest_float('mea_dropout', 0, 1)
        args.mea_l2 = trial.suggest_float('mea_l2', 1e-5, 1, log=True)

        val_dataset.reset()
        test_dataset.reset()
        train_noise_dataset.reset()
        val_noise_dataset.reset()
        
        used_budget = 0
        # 标记true val dataset并查询
        mark_dataset(args, true_model, val_dataloader)
        # 标记true test dataset并查询
        mark_dataset(args, true_model, test_dataloader)
        print(f'True val dataset size: {len(val_dataset)}')
        print(f'True test dataset size: {len(test_dataset)}')

        # 初始标记 val noise dataset并查询
        mark_dataset(args, true_model, val_noise_dataloader, int(0.2*args.initial_size))
        # 初始标记train dataset S0
        mark_dataset(args, true_model, train_noise_dataloader, int(0.8*args.initial_size))
        used_budget += args.initial_size

        loss_func = nn.CrossEntropyLoss()
        
        max_entropy = entropy(torch.ones((val_dataset.get_num_classes())))
        
        step = 0
        weighted_val_score = 0
        for it in range(args.num_iter):
            # copy model
            copy_model_type = load_model(args.copy_model)
            if args.copy_model.startswith('cnn'):
                if args.mea_dropout is not None:
                    copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[], drop_prob=args.mea_dropout).to(args.device)
                elif args.true_dataset == 'cifar':
                    copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[], drop_prob=0.2).to(args.device)
                else:
                    copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[]).to(args.device)
            elif args.copy_model.startswith('resnet'):
                copy_model = copy_model_type()

            if args.pretrain is not None:
                parms = torch.load(args.pretrain)
                del parms['fc.weight']
                del parms['fc.bias']
                if args.true_dataset == 'mnist':
                    del parms['conv_blocks.0.conv_blocks.0.weight']
                copy_model.load_state_dict(parms, strict=False)
            copy_model.to(args.device)
            # optimizer  = optim.Adam(copy_model.parameters(), lr=args.lr, weight_decay=args.mea_l2)
            if args.optimizer == 'adam':
                opti = optim.Adam
            elif args.optimizer == 'sgd':
                opti = optim.SGD
            if args.copy_model.startswith('cnn'):
                if args.lr > 0:
                    optimizer  = opti([
                        {'params': copy_model.fc.parameters()},
                        {'params': copy_model.conv_blocks.parameters(), 'weight_decay': args.train_l2}
                        ], lr=args.lr)
                else:
                    optimizer  = opti([
                        {'params': copy_model.fc.parameters()},
                        {'params': copy_model.conv_blocks.parameters(), 'weight_decay': args.train_l2}
                        ])
            elif args.copy_model.startswith('resnet'):
                optimizer  = opti([{"params":copy_model.model.layer4.parameters()},
                                        {"params":copy_model.model.fc.parameters()}], weight_decay=args.train_l2)
                

            # 训练替代模型
            copy_model.train()
            train_noise_dataset.set_state('marked')
            val_noise_dataset.set_state('marked')
            print(f'Marked Train Noise Dataset Size: {len(train_noise_dataset)}')
            print(f'Marked Val Noise Dataset Size: {len(val_noise_dataset)}')
            print(f'Used Budget: {used_budget}')
            loss_list = []
            acc_list = []
            early_stopping = EarlyStopping(patience=args.patience, verbose=True, trace_func=None)
            # early_stopping = EarlyStopping(patience=300, verbose=True, trace_func=None, delta=1.11e-5)
            for epoch in trange(args.num_epoch):
                for trX, _, idx, p in train_noise_dataloader:
                    optimizer.zero_grad()
                    trY = copy_model(trX.to(args.device))
                    label = torch.max(p.to(args.device), dim=-1, keepdim=False)[-1]
                    if args.api_retval == 'onehot':
                        # loss = loss_func(trY, label) + 5*max_entropy/entropy(torch.mean(trY, dim=0))
                        loss = loss_func(trY, label)
                    elif args.api_retval == 'softmax':
                        loss = loss_func(trY, p.to(args.device).softmax(dim=-1))
                    else:
                        raise NotImplementedError
                    loss.backward()
                    optimizer.step()
                    
                    pred = torch.max(trY, dim=-1, keepdim=False)[-1]
                    acc = pred.eq(label).cpu().numpy().mean()
                
                    loss_list.append(loss.item())
                    acc_list.append(acc)
                    # print(f'Iter: {it}\t Epoch: {epoch}\t Loss: {loss.item()}\t ACC: {acc}')
                
                # "We set aside 20% of the query budget for validation"
                val_noise_loss, val_noise_acc, val_noise_f1 = eval(args, copy_model, val_noise_dataloader, print_result=False)
                # print(val_noise_acc)
                early_stopping(val_noise_f1, np.mean(loss_list), copy_model)
                if early_stopping.early_stop:
                    break
                step += 1
            
            weighted_val_score += 1/args.num_iter * val_noise_acc
            copy_model.load_state_dict(early_stopping.best_model_parms)

            train_loss = np.array(loss_list).mean()
            train_acc = np.array(acc_list).mean()
            print('Copy model iter: {}\t Train Loss: {:.6}\t Train Agr: {:.6}\t'.format(it, train_loss, train_acc))
            print('Val Loss: {:.6}\t Val Agr: {:.2}\t Val F1: {:.2}\t'.format(val_noise_loss, val_noise_acc, val_noise_f1))
            
            # test
            print('Test on true dataset')
            test_loss, test_acc, test_f1 = eval(args, copy_model, test_dataloader)
            print('Test Loss: {:.6}\t Test Agr: {:.2}\t Test F1: {:.2}\t'.format(test_loss, test_acc, test_f1))
            
            if it == args.num_iter - 1:
                break
                
            # 使用替代模型查询剩余未标记样本标签
            copy_model.eval()
            train_noise_dataset.set_state('unmark')
            val_noise_dataset.set_state('unmark')
            
            Y = None
            Idx = []
            with torch.no_grad():
                for x, _, idx, _ in train_noise_dataloader:
                    y = copy_model(x.to(args.device))
                    if Y is None:
                        Y = y
                    else:
                        Y = torch.concat([Y, y], dim=0)
                    Idx += idx.tolist()
                
            # Active Learning策略
            b = int(0.8*args.k)
            if args.sampling_method == 'random':
                sss = RandomSelectionStrategy(b, Idx, Y)
            elif args.sampling_method == 'uncertainty':
                sss = UncertaintySelectionStrategy(b, Idx, F.softmax(Y.cpu(),dim=-1))
            elif args.sampling_method == 'certainty':
                sss = CertaintySelectionStrategy(b, Idx, F.softmax(Y.cpu(),dim=-1))
            elif args.sampling_method == 'kcenter':
                prob = train_noise_dataset.aux_data.values()
                true_points = torch.concat(list(prob), dim=0).reshape(len(prob), -1)
                sss = KCenterGreedyApproach(b, Idx, Y, true_points, args.batch_size)
            # elif args.sampling_method == 'deepfool':
            #     sss = AdversarialSelectionStrategy(args.k, Idx, Y)
            s = sss.get_subset()
            for i in s:
                train_noise_dataset.mark(i)
            lebel_dataset(args, true_model, train_noise_dataloader)
            for i in range(int(0.2*used_budget), int(0.2*(used_budget + args.k))):
                val_noise_dataset.mark(i)
            lebel_dataset(args, true_model, val_noise_dataloader)
            used_budget += args.k

        print("---Copynet trainning completed---")
        return weighted_val_score
    
    # true dataset
    val_dataset, test_dataset, val_dataloader, test_dataloader = load_true_data(args)
    if args.true_dataset not in ['agnews', 'imdb']:
        sample_shape = val_dataset.get_sample_shape()
        width, height, channels = sample_shape
        args.resize = (width, height)
    
    num_classes = val_dataset.get_num_classes()
    
    # copy dataset
    train_noise_dataset, val_noise_dataset, train_noise_dataloader, val_noise_dataloader = load_noise_data(args)
    
    # true model
    true_model_dir = os.path.join(args.path_prefix, 'saved', 
                                f'sm_{args.source_model}', 
                                f'td_{args.true_dataset}', 
                                f't_drop_{args.train_dropout}',
                                f't_l2_{args.train_l2}','true')
    if not os.path.exists(true_model_dir):
        print('Train true model first!')
    source_model_type = load_model(args.source_model)
    if args.true_dataset not in ['agnews', 'imdb']:
        if args.mea_dropout is not None:
            true_model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=args.mea_dropout)
        elif args.true_dataset == 'cifar':
            true_model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=0.2)
        else:
            true_model = source_model_type(num_classes, args.true_dataset, channels)
    else:
        vocab_size = train_noise_dataset.get_vocab_size()
        true_model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
    print(true_model)
    true_model.load_state_dict(torch.load(os.path.join(true_model_dir, 'trained_model.pth')))
    true_model = true_model.to(args.device)
    
    # 使用Optuna库进行超参数搜索
    study = optuna.create_study(direction='maximize') # 根据目标函数来决定是'minimize'还是'maximize'
    study.optimize(objective, n_trials=15) # n_trials 指的是试验次数

    # 输出最优的超参数组合和性能指标
    print('Best hyperparameters: {}'.format(study.best_params))
    print('Best performance: {:.4f}'.format(study.best_value))