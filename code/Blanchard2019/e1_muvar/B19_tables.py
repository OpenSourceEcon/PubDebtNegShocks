'''
------------------------------------------------------------------------
Create plots for slides
------------------------------------------------------------------------
'''
# Import packages
import os
import pickle
import numpy as np

# Load pickled results
cur_dir = os.path.split(os.path.abspath(__file__))[0]
output_dir = os.path.join(cur_dir, 'OUTPUT')
for H_ind in range(2):
    for risk_type_ind in range(2):
        for risk_val_ind in range(3):
            filename = ('dict_endog_' + str(H_ind) +
                        str(risk_type_ind) + str(risk_val_ind))
            print('filename=', filename)
            exec(filename + '_path = os.path.join(output_dir, \'' +
                 filename + '.pkl\')')
            exec(filename + ' = pickle.load(open(' + filename +
                 '_path, \'rb\'))')
            exec('ut_arr_' + str(H_ind) + str(risk_type_ind) +
                 str(risk_val_ind) + ' = ' + filename +
                 '[\'ut_arr\'][' + str(H_ind) + ', ' +
                 str(risk_type_ind) + ', ' + str(risk_val_ind) +
                 ', :, :, :, :]')
            print('ut_arr_' + str(H_ind) + str(risk_type_ind) +
                  str(risk_val_ind))
            exec('rbart_an_arr_' + str(H_ind) + str(risk_type_ind) +
                 str(risk_val_ind) + ' = ' + filename +
                 '[\'rbart_an_arr\'][' + str(H_ind) + ', ' +
                 str(risk_type_ind) + ', ' + str(risk_val_ind) +
                 ', :, :, :, :]')

# Solve for percent difference in average welfare matrices
avg_rtp1_size = 3
avg_rbart_size = 3

for risk_type_ind in range(2):
    for risk_val_ind in range(3):
        # ut_pctdif_1?? = np.zeros((?, ?))
        exec('ut_pctdif_1' + str(risk_type_ind) + str(risk_val_ind) +
             ' = np.zeros((avg_rtp1_size, avg_rbart_size))')
        for avgrtp1_ind in range(avg_rtp1_size):
            for avgrbart_ind in range(avg_rbart_size):
                # ut_mat_0?? = ut_arr_0??[?, ?, :, :]
                exec('ut_mat_0' + str(risk_type_ind) +
                     str(risk_val_ind) + ' = ut_arr_0' +
                     str(risk_type_ind) + str(risk_val_ind) +
                     '[avgrtp1_ind, avgrbart_ind, :, :]')
                # ut_mat_1?? = ut_arr_1??[?, ?, :, :]
                exec('ut_mat_1' + str(risk_type_ind) +
                     str(risk_val_ind) + ' = ut_arr_1' +
                     str(risk_type_ind) + str(risk_val_ind) +
                     '[avgrtp1_ind, avgrbart_ind, :, :]')
                # avg_ut_0?? = ut_mat_0??[~np.isnan(ut_mat0??)].mean()
                exec('avg_ut_0' + str(risk_type_ind) +
                     str(risk_val_ind) + ' = ut_mat_0' +
                     str(risk_type_ind) + str(risk_val_ind) +
                     '[~np.isnan(ut_mat_0' + str(risk_type_ind) +
                     str(risk_val_ind) + ')].mean()')
                # avg_ut_1?? = ut_mat_1??[~np.isnan(ut_mat1??)].mean()
                exec('avg_ut_1' + str(risk_type_ind) +
                     str(risk_val_ind) + ' = ut_mat_1' +
                     str(risk_type_ind) + str(risk_val_ind) +
                     '[~np.isnan(ut_mat_1' + str(risk_type_ind) +
                     str(risk_val_ind) + ')].mean()')
                # ut_pctdif_1??[?, ?] = (avg_ut_1?? - avg_ut_0??) /
                #                       avg_ut_0??
                exec('ut_pctdif_1' + str(risk_type_ind) +
                     str(risk_val_ind) + '[avgrtp1_ind, avgrbart_ind]' +
                     ' = (avg_ut_1' + str(risk_type_ind) +
                     str(risk_val_ind) + ' - avg_ut_0' +
                     str(risk_type_ind) + str(risk_val_ind) +
                     ') / avg_ut_0' + str(risk_type_ind) +
                     str(risk_val_ind))
        # print('ut_pctdif_1??')
        exec('print(\'ut_pctdif_1' + str(risk_type_ind) +
             str(risk_val_ind) + ' for Cobb-Douglas, mu variable\')')
        # print(ut_pctdif_1??)
        exec('print(ut_pctdif_1' + str(risk_type_ind) +
             str(risk_val_ind) + ')')
        # print('')
        exec('print(\'\')')
