去掉了            
if args.n_bits < 8:
image = torch.floor(image / 2 ** (8 - args.n_bits))