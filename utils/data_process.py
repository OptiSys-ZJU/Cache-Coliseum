import pandas as pd
import os
import csv

if __name__ == '__main__':
    name_dict = {
        'LRU': 'LRU',

        'OracleLogDis': {
            'Predict[Belady]': 'ftp',
            'PredMark[Belady]': 'predmarker',
            'LMarker[Belady]': 'lmarker',
            'LNonMarker[Belady]': 'lnonmarker',
            'FollowerRobust[OracleState]': "fr",
            'Guard[Belady]-f-pred-no-relax': 'guardftp0',
            'Guard[Belady]-f-pred-relax-times-5': 'guardftp5',
            'CombDet[Predict[Belady], Marker]': 'blindoracledftp',
            'CombineRandom[Predict[Belady], Marker]': 'blindoraclerftp',
        },

        'OracleDis': {
            'Predict[Belady]': 'ftp',
            'PredMark[Belady]': 'predmarker',
            'LMarker[Belady]': 'lmarker',
            'LNonMarker[Belady]': 'lnonmarker',
            'FollowerRobust[OracleState]': "fr",
            'Guard[Belady]-f-pred-no-relax': 'guardftp0',
            'Guard[Belady]-f-pred-relax-times-5': 'guardftp5',
            'CombDet[Predict[Belady], Marker]': 'blindoracledftp',
            'CombineRandom[Predict[Belady], Marker]': 'blindoraclerftp',
        },

        'OracleBin': {
            'Predict[FBP]': 'fbp',
            'Mark0[FBP]': 'mark0',
            'Mark&Predict[OraclePhase]': 'markpred',
            'CombDet[Predict[FBP], Marker]': 'blindoracledfbp',
            'CombineRandom[Predict[FBP], Marker]': 'blindoraclerfbp',
            'Guard[FBP]-f-pred-no-relax': 'guardfbp0',
            'Guard[FBP]-f-pred-relax-times-5': 'guardfbp5',
        },

        'Parrot': {
            'Marker': 'Marker',
            'Predict[Parrot]': 'FTP',
            'PredMark[Parrot]': 'PredMark',
            'LMarker[Parrot]': 'LMarker',
            'LNonMarker[Parrot]': 'LNonMarker',
            'CombDet[Predict[Parrot], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[Parrot], Marker]': 'BlindOracleR',
            'FollowerRobust[ParrotState]': 'F&R',
            'Guard[Parrot]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[Parrot]-f-pred-relax-times-5': 'Guard&FTP5',
        },

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
    mode = 'Parrot'
    root_dir_path = 'stat'
    res_csv_path = 'plot_res'
    res_dict = {}
    for dirpath in os.listdir(root_dir_path):
        dataset = dirpath
        # if dataset == 'brightkite' or dataset == 'citi':
        #     continue
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
            elif mode == 'OracleLogDis':
                this_dataset_plot_path = os.path.join(res_csv_path, dataset, 'logdis')
                if not os.path.exists(this_dataset_plot_path):
                    os.makedirs(this_dataset_plot_path)
                dir_path = os.path.join(full_path, '1')                
                this_path = os.path.join(dir_path, 'logdis.csv')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    for name in result_dict:
                        if name in name_dict[mode]:
                            alg_name = name_dict[mode][name]
                            res_path = os.path.join(this_dataset_plot_path, f'{alg_name}.csv')
                            tuple_list = []
                            for noise, value in result_dict[name].items():
                                if noise.startswith('LogDis-'):
                                    this_x = noise.split('-')[1]
                                    this_y = value.split('/')[1]
                                    tuple_list.append((this_x, this_y))
                            with open(res_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['x', 'y'])
                                for tup in tuple_list:
                                    writer.writerow(tup)
                            print(f"Data has been written to {res_path}")
            elif mode == 'OracleDis':
                this_dataset_plot_path = os.path.join(res_csv_path, dataset, 'dis')
                if not os.path.exists(this_dataset_plot_path):
                    os.makedirs(this_dataset_plot_path)
                dir_path = os.path.join(full_path, '1')                
                this_path = os.path.join(dir_path, 'dis.csv')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    for name in result_dict:
                        if name in name_dict[mode]:
                            alg_name = name_dict[mode][name]
                            res_path = os.path.join(this_dataset_plot_path, f'{alg_name}.csv')
                            tuple_list = []
                            for noise, value in result_dict[name].items():
                                if noise.startswith('Dis-'):
                                    this_x = noise.split('-')[1]
                                    this_y = value.split('/')[1]
                                    tuple_list.append((this_x, this_y))
                            with open(res_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['x', 'y'])
                                for tup in tuple_list:
                                    writer.writerow(tup)
                            print(f"Data has been written to {res_path}")
            elif mode == 'OracleBin':
                this_dataset_plot_path = os.path.join(res_csv_path, dataset, 'bin')
                if not os.path.exists(this_dataset_plot_path):
                    os.makedirs(this_dataset_plot_path)
                dir_path = os.path.join(full_path, '1')                
                this_path = os.path.join(dir_path, 'bin.csv')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    for name in result_dict:
                        if name in name_dict[mode]:
                            alg_name = name_dict[mode][name]
                            res_path = os.path.join(this_dataset_plot_path, f'{alg_name}.csv')
                            tuple_list = []
                            for noise, value in result_dict[name].items():
                                if noise.startswith('Bin-'):
                                    this_x = noise.split('-')[1]
                                    this_y = value.split('/')[1]
                                    tuple_list.append((this_x, this_y))
                            with open(res_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['x', 'y'])
                                for tup in tuple_list:
                                    writer.writerow(tup)
                            print(f"Data has been written to {res_path}")
            else:
                dir_path = os.path.join(full_path, '1')                
                # this_path = os.path.join(dir_path, 'pleco_popu_pleco-bin.csv')
                this_path = os.path.join(dir_path, 'parrot.csv')
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

            
