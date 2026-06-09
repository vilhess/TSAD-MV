import torch 

def get_mask(test_signal, seed, missing_rate, strategy="mcar"):
    torch.manual_seed(seed)

    if strategy == "mcar":
        mask = torch.rand(test_signal.shape) < missing_rate
        return mask.float()

    elif strategy == "seq_block":
        n_missing = int(test_signal.numel() * missing_rate)
        n_missing_per_variate = int(n_missing / test_signal.shape[1])
        mask = torch.zeros(test_signal.shape)
        for i in range(test_signal.shape[1]):
            n_sub_block = torch.randint(6, 12, (1,)).item()
            block_size = int(n_missing_per_variate // n_sub_block)
            for _ in range(n_sub_block):
                start_idx = torch.randint(0, test_signal.shape[0] - block_size, (1,)).item()
                mask[start_idx:start_idx + block_size, i] = 1

        return mask.float()

    else:
        raise ValueError("Invalid strategy. Choose 'mcar' or 'seq_block'.")