if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils.utils import load_config, load_img
    from utils.ifm import IFM

    ap = ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/example.cfg', help='Config file for the water type and coefficients')
    args = ap.parse_args()

    config = load_config(args)

    print(f'Loading image: {config["input_img"]}')
    img = load_img(config['input_img'])

    if config.get('SUID', None):
        # TODO
        print('Using the SUID implementation')
        pass
    else:
        print('Using the underwater IFM implementation')
        ifm = IFM(out_dir=config['output_dir'],
                  coeff_data_table=config['coeff_data_table'],
                  wtypes=config['wtype'],
                  vdepths=config['vdepths'],
                  fscatter=config['fscatter'],
                  g=config['g'],
                  mu=config['mu'],
                  inhomog=config['inhomog'],
                  grf_min=config['grf_min'],
                  grf_max=config['grf_max'],
                  camera=config['camera']
                )
        
        ifm.generate(img, config['z_min'], config['z_max'], config['gamma'])

