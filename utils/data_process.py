import pandas as pd
import os

if __name__ == '__main__':
    name_dict = {
        'LRU': 'LRU',
        'PLECO': {
            'Marker': 'Marker',
            'Predict[PLECO]': 'FTP',
            'PredMark[PLECO]': 'PredMark',
            'LMarker[PLECO]': 'LMarker',
            'LNonMarker[PLECO]': 'LNonMarker',
            'CombDet[Predict[PLECO], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[PLECO], Marker]': 'BlindOracleR',
            'FollowerRobust[PLECOState]': 'F&R',
            'Guard[PLECO]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[PLECO]-f-pred-relax-times-5': 'Guard&FTP5',
        },

        'POPU': {
            'Marker': 'Marker',
            'Predict[POPU]': 'FTP',
            'PredMark[POPU]': 'PredMark',
            'LMarker[POPU]': 'LMarker',
            'LNonMarker[POPU]': 'LNonMarker',
            'CombDet[Predict[POPU], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[POPU], Marker]': 'BlindOracleR',
            'FollowerRobust[POPUState]': 'F&R',
            'Guard[POPU]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[POPU]-f-pred-relax-times-5': 'Guard&FTP5',
        },

        'PLECOBin': {
            'Marker': 'Marker',
            'Predict[PLECOBin]': 'FBP',
            'Mark0[PLECOBin]': 'Mark0',
            'CombDet[Predict[PLECOBin], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[PLECOBin], Marker]': 'BlindOracleR',
            'Guard[PLECOBin]-f-pred-no-relax': 'Guard&FBP0',
            'Guard[PLECOBin]-f-pred-relax-times-5': 'Guard&FBP5',
        },

        'GBMBin': {
            'Marker': 'Marker',
            'Predict[GBMBin]': 'FBP',
            'Mark0[GBMBin]': 'Mark0',
            'CombDet[Predict[GBMBin], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[GBMBin], Marker]': 'BlindOracleR',
            'Guard[GBMBin]-f-pred-no-relax': 'Guard&FBP0',
            'Guard[GBMBin]-f-pred-relax-times-5': 'Guard&FBP5',
        },

    }
    mode = 'GBMBin'
    root_dir_path = 'stat'
    res_dict = {}
    for dirpath in os.listdir(root_dir_path):
        dataset = dirpath
        if dataset == 'brightkite' or dataset == 'citi':
            continue
        full_path = os.path.join(root_dir_path, dirpath)
        if os.path.isdir(full_path):
            if mode == 'GBMBin':
                res_dict[dataset] = {}
                for frac_path in os.listdir(full_path):
                    frac = float(frac_path)
                    this_path = os.path.join(full_path, frac_path, 'gbm.csv')
                    if os.path.exists(this_path):
                        df = pd.read_csv(this_path)
                        result_dict = df.set_index('Name').T.to_dict('dict')

                        lru = result_dict['LRU']['Competitive Ratio']

                        for name in result_dict:
                            cr = result_dict[name]['Competitive Ratio']
                            res = (cr-1)/(lru-1)
                            if name in name_dict[mode]:
                                res_name = name_dict[mode][name]
                                if res_name not in res_dict[dataset]:
                                    res_dict[dataset][res_name] = []
                                res_dict[dataset][res_name].append((frac, (cr-1)/(lru-1)))

            else:
                dir_path = os.path.join(full_path, '1')                
                this_path = os.path.join(dir_path, 'pleco_popu_pleco-bin.csv')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')

                    lru = result_dict['LRU']['Competitive Ratio']

                    for name in result_dict:
                        cr = result_dict[name]['Competitive Ratio']
                        if name in name_dict[mode]:
                            res_name = name_dict[mode][name]
                            if res_name not in res_dict:
                                res_dict[res_name] = []
                            res_dict[res_name].append((dataset, (cr-1)/(lru-1)))

    if mode == 'GBMBin':
        for dataset, d in sorted(res_dict.items()):
            print(dataset)
            for name, l in d.items():
                print(name)
                sorted_list = sorted(l, key=lambda x: x[0])
                for t in sorted_list:
                    print(f'({t[0]},{t[1]})')
    else:
        for name, l in res_dict.items():
            sorted_list = sorted(l, key=lambda x: x[0])
            print(name)
            for t in sorted_list:
                print(f'({t[0]},{t[1]})')

            
