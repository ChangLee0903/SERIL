import torch
from dataset import SpeechDataset

channel = 0
feat_list = [
    {'feat_type': 'complx', 'channel': channel},
    {'feat_type': 'linear', 'channel': channel},
    {'feat_type': 'phase', 'channel': channel}
]


def save_model(self, save_type=None):
    all_states = {
        'Upstream': self.upstream_model.state_dict() if self.args.fine_tune else None,
        'Downstream': self.downstream_model.state_dict(),
        'Optimizer': self.optimizer.state_dict(),
        'Global_step': self.global_step,
        'Settings': {
            'Config': self.config,
            'Paras': self.args,
        },
    }

    def check_ckpt_num(directory):
        ckpts = glob.glob(f'{directory}/states-*.ckpt')
        if len(ckpts) >= self.rconfig['max_keep']:
            ckpts = sorted(ckpts, key=lambda pth: int(
                pth.split('-')[-1].split('.')[0]))
            for ckpt in ckpts[:len(ckpts) - self.rconfig['max_keep']]:
                os.remove(ckpt)

    save_dir = self.expdir if save_type is None else f'{self.expdir}/{save_type}'
    os.makedirs(save_dir, exist_ok=True)
    check_ckpt_num(save_dir)
    model_path = f'{save_dir}/states-{self.global_step}.ckpt'
    torch.save(all_states, model_path)


def get_dataloader(args, config, idx):
    def collate_fn(samples):
        niy_samples = [s[0] for s in samples]
        cln_samples = [s[1] for s in samples]
        lengths = torch.LongTensor([len(s[0]) for s in samples])

        niy_samples = torch.nn.utils.rnn.pad_sequence(
            niy_samples, batch_first=True)
        cln_samples = torch.nn.utils.rnn.pad_sequence(
            cln_samples, batch_first=True)
        return lengths, niy_samples.transpose(-1, -2).contiguous(), cln_samples.transpose(-1, -2).contiguous()

    noisy_list = config['noisy'][idx]
    clean_list = config['clean'][idx]

    dataloader = torch.utils.data.DataLoader(SpeechDataset(
            noisy_list, clean_list), config['batch_size'], collate_fn=collate_fn, num_workers=args.n_jobs, shuffle=True)

    return dataloader