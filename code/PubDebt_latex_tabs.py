'''
------------------------------------------------------------------------
This module contains the functions used in the Public Debt and Negative
Shocks paper for creating LaTeX tables
------------------------------------------------------------------------
'''

# Import packages
from numba import jit

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


@jit
def print_latex_tab_med(Hbar_vec, k20_vec, four_matrices, decimals=3):
    '''
    --------------------------------------------------------------------
    This function returns a multi-line string of LaTeX table output
    derived from original input matrices. Initial values relative to
    median values
    --------------------------------------------------------------------
    INPUTS:
    Hbar_vec      = (Hbar_size,) vector, values of Hbar
    k20_vec       = (k20_size,) vector, values of k_{2,0}
    four_matrices = length 4 tuple, four matrices of statistics

    med_wt_Hk         = (Hbar_size, k20_size) matrix, median values of
                        w_t
    med_k2t_Hk        = (Hbar_size, k20_size) matrix, median values of
                        k_{2,t}
    med_Hbar_wtmed_Hk = (Hbar_size, k20_size) matrix, Hbar as percent of
                        median values of w_t
    med_k20_k2tmed_Hk = (Hbar_size, k20_size) matrix, k_{2,0} as percent
                        of median values of k_{2,t}
    --------------------------------------------------------------------
    '''
    med_wt_Hk, med_k2t_Hk, med_Hbar_wtmed_Hk, med_k20_k2tmed_Hk = \
        four_matrices
    rows, cols = med_wt_Hk.shape
    mat_string = r'\hline\hline ' + '\n'
    for k20_val in k20_vec:
        mat_string += \
            (r'& \multicolumn{2}{c}{$k_{2,0}=' + str(k20_val) + r'$} ')
    mat_string += r'\\ \cline{2-7} ' + '\n'
    for col in range(cols):
        mat_string += r'& $w_{med}$ & $k_{med}$ '
    mat_string += r'\\ ' + '\n'
    for col in range(cols):
        mat_string += r'& $\bar{H}/w_{med}$ & $k_{2,0}/k_{med}$ '
    mat_string += r'\\ ' + '\n'

    for row in range(rows):
        mat_string += r'\hline ' + '\n'
        mat_string += (r'\multirow{2}{*}{$\bar{H}=' +
                       '{:.2f}'.format(Hbar_vec[row]) + r'$} ' + '\n')
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(med_wt_Hk[row, col],
                                              prec=decimals) + r' & ' +
                 '{:.{prec}f}'.format(med_k2t_Hk[row, col],
                                      prec=decimals) + r' ')
        mat_string += r'\\ ' + '\n'
        for col in range(cols):
            mat_string += \
                (r'& ' +
                 '{:.{prec}f}'.format(med_Hbar_wtmed_Hk[row, col],
                                      prec=decimals) + r' & ' +
                 '{:.{prec}f}'.format(med_k20_k2tmed_Hk[row, col],
                                      prec=decimals) + r' ')
        mat_string += r'\\ ' + '\n'
    mat_string += r'\hline\hline'

    print('\n BEGIN MEDIAN TABLE \n')
    print(mat_string + '\n')

    return mat_string


@jit
def print_latex_tab_shut(Hbar_vec, k20_vec, per_matrices, cdf_matrices,
                         decimals_avg=1, decimals_cdf=3):
    '''
    --------------------------------------------------------------------
    This function returns a multi-line string of LaTeX table output
    derived from original input matrices. Periods to shut down
    simulation statistics
    --------------------------------------------------------------------
    INPUTS:
    Hbar_vec     = (Hbar_size,) vector, values of Hbar
    k20_vec      = (k20_size,) vector, values of k_{2,0}
    per_matrices = length 4 tuple, four matrices of statistics
    cdf_matrices = length 4 tuple, four matrices of statistics
    --------------------------------------------------------------------
    '''
    min_per2GO_Hk, med_per2GO_Hk, avg_per2GO_Hk, max_per2GO_Hk = \
        per_matrices
    min_cdf2GO_Hk, med_cdf2GO_Hk, avg_cdf2GO_Hk, max_cdf2GO_Hk = \
        cdf_matrices
    rows, cols = avg_per2GO_Hk.shape
    mat_string = r'\hline\hline ' + '\n'
    mat_string += '& '
    for k20_val in k20_vec:
        mat_string += \
            (r'& \multicolumn{2}{c}{$k_{2,0}=' + str(k20_val) + r'$} ')
    mat_string += r'\\ \cline{3-8} ' + '\n'
    mat_string += '& '
    for col in range(cols):
        mat_string += r'& Periods & CDF '
    mat_string += r'\\ ' + '\n'

    for row in range(rows):
        mat_string += r'\hline ' + '\n'
        mat_string += (r'\multirow{4}{*}{$\bar{H}=' +
                       '{:.2f}'.format(Hbar_vec[row]) + r'$} ' + '\n')
        mat_string += '& min '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.0f}'.format(min_per2GO_Hk[row, col]) +
                 r' & ' + '{:.{prec}f}'.format(min_cdf2GO_Hk[row, col],
                                               prec=decimals_cdf) +
                 r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& med '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.0f}'.format(med_per2GO_Hk[row, col]) +
                 r' & ' + '{:.{prec}f}'.format(med_cdf2GO_Hk[row, col],
                                               prec=decimals_cdf) +
                 r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& mean '
        for col in range(cols):
            if row == 0:
                mat_string += \
                    (r'& ' + '{:.0f}'.format(avg_per2GO_Hk[row, col]) +
                     r' & ' +
                     '{:.{prec}f}'.format(avg_cdf2GO_Hk[row, col],
                                          prec=decimals_cdf) + r' ')
            else:
                mat_string += \
                    (r'& ' +
                     '{:.{prec}f}'.format(avg_per2GO_Hk[row, col],
                                          prec=decimals_avg) + r' & ' +
                     '{:.{prec}f}'.format(avg_cdf2GO_Hk[row, col],
                                          prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& max '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.0f}'.format(max_per2GO_Hk[row, col]) +
                 r' & ' + '{:.{prec}f}'.format(max_cdf2GO_Hk[row, col],
                                               prec=decimals_cdf) +
                 r' ')
        mat_string += r'\\ ' + '\n'
    mat_string += r'\hline\hline'

    print('\n BEGIN SHUTDOWN TABLE \n')
    print(mat_string + '\n')

    return mat_string


@jit
def print_latex_tab_riskl(Hbar_vec, k20_vec, rate_matrices,
                          cdf_matrices, decimals_pct=2, decimals_cdf=3):
    '''
    --------------------------------------------------------------------
    This function returns a multi-line string of LaTeX table output
    derived from original input matrices. Periods to shut down
    simulation statistics
    --------------------------------------------------------------------
    INPUTS:
    Hbar_vec     = (Hbar_size,) vector, values of Hbar
    k20_vec      = (k20_size,) vector, values of k_{2,0}
    per_matrices = length 5 tuple, four matrices of statistics
    cdf_matrices = length 5 tuple, four matrices of statistics
    --------------------------------------------------------------------
    '''
    (t0_rbart_an_Hk, min_rbart_an_Hk, med_rbart_an_Hk, avg_rbart_an_Hk,
        max_rbart_an_Hk) = rate_matrices
    (t0_rbartcdf_an_Hk, min_rbartcdf_an_Hk, med_rbartcdf_an_Hk,
        avg_rbartcdf_an_Hk, max_rbartcdf_an_Hk) = cdf_matrices
    rows, cols = avg_rbart_an_Hk.shape
    mat_string = r'\hline\hline ' + '\n'
    mat_string += '& '
    for k20_val in k20_vec:
        mat_string += \
            (r'& \multicolumn{2}{c}{$k_{2,0}=' + str(k20_val) + r'$} ')
    mat_string += r'\\ \cline{3-8} ' + '\n'
    mat_string += '& '
    for col in range(cols):
        mat_string += r'& $\bar{r}_{t,an}$ & CDF '
    mat_string += r'\\ ' + '\n'

    for row in range(rows):
        mat_string += r'\hline ' + '\n'
        mat_string += (r'\multirow{5}{*}{$\bar{H}=' +
                       '{:.2f}'.format(Hbar_vec[row]) + r'$} ' + '\n')
        mat_string += r'& $t=0$ '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(t0_rbart_an_Hk[row, col],
                                              prec=decimals_pct) +
                 r'\% & ' +
                 '{:.{prec}f}'.format(t0_rbartcdf_an_Hk[row, col],
                                      prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& min '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(min_rbart_an_Hk[row, col],
                                              prec=decimals_pct) +
                 r'\% & ' +
                 '{:.{prec}f}'.format(min_rbartcdf_an_Hk[row, col],
                                      prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& med '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(med_rbart_an_Hk[row, col],
                                              prec=decimals_pct) +
                 r'\% & ' +
                 '{:.{prec}f}'.format(med_rbartcdf_an_Hk[row, col],
                                      prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& mean '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(avg_rbart_an_Hk[row, col],
                                              prec=decimals_pct) +
                 r'\% & ' +
                 '{:.{prec}f}'.format(avg_rbartcdf_an_Hk[row, col],
                                      prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
        mat_string += '& max '
        for col in range(cols):
            mat_string += \
                (r'& ' + '{:.{prec}f}'.format(max_rbart_an_Hk[row, col],
                                              prec=decimals_pct) +
                 r'\% & ' +
                 '{:.{prec}f}'.format(max_rbartcdf_an_Hk[row, col],
                                      prec=decimals_cdf) + r' ')
        mat_string += r'\\ ' + '\n'
    mat_string += r'\hline\hline'

    print('\n BEGIN RISKLESS ASSET TABLE \n')
    print(mat_string + '\n')

    return mat_string
