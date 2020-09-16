import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../../pyorb/')
sys.path.insert(0, '../../../pyorb/examples/affine_advection_cpp')

import affine_advection as aap
import pyorb_core.rb_library.rb_manager as rm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_handler as ph
from pylab import *


resdir = 'results/'
resdir = 'results_inputequaloutput/'
# runs = 1
# runs_range = [2,3,4,5,6,7,8,9]
runs_range = range(10)
runs = len(runs_range)

def dir_name( inr, outr, tol, ns, ep ):
    return 'ni' + str( inr ) + '_' \
           'no' + str( outr ) + '_' \
           'rbtol' + str( tol ) + '_' \
           'ns' + str( ns ) + '_' \
           'ntconstant_' + \
           'ep' + str( ep )

def generate_dir_name( inr, outr, tol, ns, ep ):
    dirname = dir_name( inr, outr, tol, ns, ep )
    if os.path.exists( resdir + dirname ):
        return dirname
    print( dirname + ' does not exist!' )
    return False    

def plot_histories( network_type, varname, dirname ):
    plt.figure()
    f, (ax1, ax2) = plt.subplots( 1, 2, figsize = (15,4) )
    mean = 0
    mean_val = 0
    for i in runs_range:
        locres = np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/' + varname + '.npy' )
        locres_val = np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/val_' + varname + '.npy' )
        mean = mean + locres
        mean_val = mean_val + locres_val
        ax1.semilogy( locres )
    mean = mean / runs
    mean_val = mean_val / runs
    ax2.semilogy( mean )
    ax2.semilogy( mean_val )
    ax2.legend( ('train', 'validation') )
    ax1.set_title( network_type + ': results on train, different runs' )
    ax2.set_title( network_type + ': average on 5 runs' )
    ax1.set_xlabel( 'epochs' )
    ax2.set_xlabel( 'epochs' )
    ax1.set_ylabel( varname )
    ax2.set_ylabel( varname )
    
def plot_histories_loss_mae( network_type, dirname ):
    plt.figure()
    varname = 'mean_absolute_error'
    f, (ax1, ax2) = plt.subplots( 1, 2, figsize = (15,4) )
    mean = 0
    mean_val = 0
    for i in runs_range:
        locres = np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/' + varname + '.npy' )
        locres_val = np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/val_' + varname + '.npy' )
        mean = mean + locres
        mean_val = mean_val + locres_val
        ax1.semilogy( np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/loss.npy' ) )
    mean = mean / runs
    mean_val = mean_val / runs
    ax2.semilogy( mean )
    ax2.semilogy( mean_val )
    ax2.legend( ('train', 'validation') )
    ax1.set_xlim( [0,500] )
    ax2.set_xlim( [0,500] )
    # ax1.set_title( network_type + ': results on train, different runs' )
    # ax2.set_title( network_type + ': average on 5 runs' )
    ax1.set_xlabel( 'epochs' )
    ax2.set_xlabel( 'epochs' )
    ax1.set_ylabel( 'loss' )
    ax2.set_ylabel( 'mean absolute error' )
    print( "Saving " + resdir + dirname  + '/loss_mae.pdf')
    plt.savefig( resdir + dirname + '/loss_mae.pdf' )
    
def find_best( values ):
    nvalues = np.array( values )
    best = np.argmin( nvalues )
    
    retlist = []
    for i in range( nvalues.size ):
        if i == best:
            retlist = retlist + [' <------']
        else:
            retlist = retlist + ['']
    
    return retlist
    
def plot_prediction_parameters( newtork_type, y, dirname ):
    plt.figure()
    f, (ax1,ax2)= plt.subplots(1, 2, figsize = (15,4) )
    mean = 0
    for i in runs_range:
        test = np.load( resdir + dirname + '/' + newtork_type + '/' + str(i) + '/test.npy' )
        mean = mean + test
        ax1.scatter( y[:,-2], test[:,-2] )
        ax2.scatter( y[:,-1], test[:,-1] )
        
    ax1.set_ylim( [0,1] )
    ax1.set_xlim( [0,1] )
    ax2.set_ylim( [0,1] )
    ax2.set_xlim( [0,1] )
    ax1.set_title( newtork_type + ': prediction on diffusivity, different runs' )
    ax2.set_title( newtork_type + ': prediction on angle, different runs' )
    ax1.set_xlabel( 'real parameter' )
    ax2.set_xlabel( 'real parameter' )
    ax1.set_ylabel( 'predicted parameter' )
    ax2.set_ylabel( 'predicted parameter' )
    
    plt.figure()
    f, (ax1,ax2)= plt.subplots(1, 2, figsize = (15,4) )
    
    mean = mean / runs
    ax1.scatter( y[:,-2], mean[:,-2] )
    ax2.scatter( y[:,-1], mean[:,-1] )
    ax1.set_ylim( [0,1] )
    ax1.set_xlim( [0,1] )
    ax2.set_ylim( [0,1] )
    ax2.set_xlim( [0,1] )
    ax1.set_title( newtork_type + ': prediction on diffusivity on average' )
    ax2.set_title( newtork_type + ': prediction on angle on average' )
    ax1.set_xlabel( 'real parameter' )
    ax2.set_xlabel( 'real parameter' )
    ax1.set_ylabel( 'predicted parameter' )
    ax2.set_ylabel( 'predicted parameter' )
    
def comparison_prediction_parameters( y, dirname ):
    plt.figure()
    f, (ax1,ax2)= plt.subplots(1, 2, figsize = (15,4) )
    mean_pde = 0
    mean_tf = 0
    mean_tf_mu = 0
    for i in runs_range:
        test_pde = np.load( resdir + dirname + '/' + 'pde' + '/' + str(i) + '/test.npy' )
        test_tf = np.load( resdir + dirname + '/' + 'tf' + '/' + str(i) + '/test.npy' )
        test_tf_mu = np.load( resdir + dirname + '/' + 'tf_mu' + '/' + str(i) + '/test.npy' )
        mean_pde = mean_pde + test_pde
        mean_tf = mean_tf + test_tf
        mean_tf_mu = mean_tf_mu + test_tf_mu        

    mean_pde = mean_pde / runs
    mean_tf = mean_tf / runs
    mean_tf_mu = mean_tf_mu / runs

    ax1.scatter( y[:,-2], mean_pde[:,-2] )
    ax1.scatter( y[:,-2], mean_tf[:,-2] )
    ax1.scatter( y[:,-2], mean_tf_mu[:,0] )
    
    mpde = np.mean( np.abs(y[:,-2] - mean_pde[:,-2] ) )
    mtf = np.mean( np.abs(y[:,-2] - mean_tf[:,-2] ) )
    mtf_mu = np.mean( np.abs(y[:,-2] - mean_tf_mu[:,0] ) )
    
    best = find_best( [mpde, mtf, mtf_mu] )
    
    print('Average error on diffusivity (pde): \t' + str(mpde) + best[0] )
    print('Average error on diffusivity (tf): \t' + str(mtf) + best[1] )
    print('Average error on diffusivity (tf_mu): \t' + str(mtf_mu) + best[2] )

    ax2.scatter( y[:,-1], mean_pde[:,-1] )
    ax2.scatter( y[:,-1], mean_tf[:,-1] )
    ax2.scatter( y[:,-1], mean_tf_mu[:,1] )
    
    mpde = np.mean( np.abs(y[:,-1] - mean_pde[:,-1] ) )
    mtf = np.mean( np.abs(y[:,-1] - mean_tf[:,-1] ) )
    mtf_mu = np.mean( np.abs(y[:,-1] - mean_tf_mu[:,1] ) )
    
    best = find_best( [mpde, mtf, mtf_mu] )

    print('Average error on angle (pde): \t\t' + str( mpde ) + best[0] )
    print('Average error on angle (tf): \t\t' + str( mtf ) + best[1] )
    print('Average error on angle (tf_mu): \t' + str( mtf_mu ) + best[2] )

    ax1.legend( ('pde','tf','tf_mu') )
    ax2.legend( ('pde','tf','tf_mu') )
    ax1.set_ylim( [0,1] )
    ax1.set_xlim( [0,1] )
    ax2.set_ylim( [0,1] )
    ax2.set_xlim( [0,1] )
    ax1.set_title( 'Comparison of prediction on diffusivity on average' )
    ax2.set_title( 'Comparison of prediction on angle on average' )
    ax1.set_xlabel( 'real parameter' )
    ax2.set_xlabel( 'real parameter' )
    ax1.set_ylabel( 'predicted parameter' )
    ax2.set_ylabel( 'predicted parameter' )
    
def scatterplot_of_errors( y, dirname ):
    fig = plt.figure( figsize = (15,4) )
    ax1 = fig.add_subplot(121, projection='3d')
    # ax1 = fig.add_subplot(111)

    femcoord = np.load( resdir + dirname + '/cut_fem_output_coordinates.npy' )
    ycut = y[:,femcoord]
    mean_pde = 0
    mean_tf = 0
    mean_loc = 0
    for i in runs_range:
        test_pde = np.load( resdir + dirname + '/' + 'pde' + '/' + str(i) + '/test.npy' )
        test_tf = np.load( resdir + dirname + '/' + 'tf' + '/' + str(i) + '/test.npy' )
        test_loc = np.load( resdir + dirname + '/' + 'tf_loc' + '/' + str(i) + '/test.npy' )
        mean_pde = mean_pde + test_pde
        mean_tf = mean_tf + test_tf
        mean_loc = mean_loc + test_loc

        err_pde = np.abs( ycut - test_pde[:,:-2] )
        
        ntest = y.shape[0]
        
        single_errs = np.zeros( (ntest,1) )
        for j in range(ntest):
            single_errs[j] = np.sqrt(np.dot(err_pde[j,:],err_pde[j,:])) / np.sqrt(np.dot(ycut[j,:],ycut[j,:]))
        ax1.scatter( y[:,-1], y[:,-2], single_errs )
        
    mean_pde = mean_pde / runs
    mean_tf = mean_tf / runs
    mean_loc = mean_loc / runs
   
    err_pde = np.abs( ycut - mean_pde[:,:-2] )
    err_tf = np.abs( ycut - mean_tf[:,:-2] )
    err_loc = np.abs( ycut - mean_loc )
       
    single_errs_pde = np.zeros( (ntest,1) )
    single_errs_tf = np.zeros( (ntest,1) )
    single_errs_loc = np.zeros( (ntest,1) )
    for j in range(ntest):
        single_errs_pde[j] = np.sqrt(np.dot(err_pde[j,:],err_pde[j,:])) / np.sqrt(np.dot(ycut[j,:],ycut[j,:]))
        single_errs_tf[j] = np.sqrt(np.dot(err_tf[j,:],err_tf[j,:])) / np.sqrt(np.dot(ycut[j,:],ycut[j,:]))
        single_errs_loc[j] = np.sqrt(np.dot(err_loc[j,:],err_loc[j,:])) / np.sqrt(np.dot(ycut[j,:],ycut[j,:]))
        
    mpde = np.mean(single_errs_pde)
    mtf = np.mean(single_errs_tf)
    mtf_loc = np.mean(single_errs_loc)
        
    best = find_best( [mpde,mtf,mtf_loc] )
    print('Average error on output (pde): \t\t' + str(mpde) + best[0] )
    print('Average error on output (tf): \t\t' + str(mtf) + best[1] )
    print('Average error on output (tf_loc): \t' + str(mtf_loc) + best[2] )

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter( y[:,-2], y[:,-1], single_errs_pde )
    ax2.scatter( y[:,-2], y[:,-1], single_errs_tf )
    ax2.scatter( y[:,-2], y[:,-1], single_errs_loc )

    ax2.legend( ('pde','tf','tf_loc') )
    ax1.set_ylim( [0,1] )
    ax1.set_xlim( [0,1] )
    ax2.set_ylim( [0,1] )
    ax2.set_xlim( [0,1] )
    ax1.set_title( 'Errors on u(x), different runs' )
    ax2.set_title( 'Comparison of errors on u(x) on average' )
    ax1.set_xlabel( 'diffusivity' )
    ax1.set_ylabel( 'angle' )
    ax1.set_zlabel( 'normalized error' )
    ax2.set_xlabel( 'diffusivity' )
    ax2.set_ylabel( 'angle' )
    ax2.set_zlabel( 'normalized error' )
    
def print_rt_information( dirname ):
    mean_pde = 0
    mean_tf = 0
    mean_tf_mu = 0
    mean_tf_loc = 0
    for i in runs_range:
        rt_pde = np.loadtxt( resdir + dirname + '/' + 'pde' + '/' + str(i) + '/rt.txt' )
        rt_tf = np.loadtxt( resdir + dirname + '/' + 'tf' + '/' + str(i) + '/rt.txt' )
        rt_tf_mu = np.loadtxt( resdir + dirname + '/' + 'tf_mu' + '/' + str(i) + '/rt.txt' )
        rt_tf_loc = np.loadtxt( resdir + dirname + '/' + 'tf_loc' + '/' + str(i) + '/rt.txt' )
        
        mean_pde = mean_pde + rt_pde
        mean_tf = mean_tf + rt_tf
        mean_tf_mu = mean_tf_mu + rt_tf_mu
        mean_tf_loc = mean_tf_loc + rt_tf_loc
    
    print( "Average running time for training (pde): \t" + str( mean_pde/runs ) )
    print( "Average running time for trainissng (tf): \t" + str( mean_tf/runs ) )
    print( "Average running time for training (tf_mu): \t" + str( mean_tf_mu/runs ) )
    print( "Average running time for training (tf_loc): \t" + str( mean_tf_loc/runs ) )
    
def compute_H1_error( y_complete, rb_tol, network_type, dirname ):
    plt.figure()
    f, (ax1, ax2) = plt.subplots( 1, 2, figsize = (15,5) )
    Y = y_complete[:,:-2]
    mu0_min = 0.5; mu0_max = 10.
    mu1_min = 0; mu1_max = 30.
    
    param_min = np.array([mu0_min, mu1_min])
    param_max = np.array([mu0_max, mu1_max])
    
    my_parameter_handler = ph.Parameter_handler( )
    my_parameter_handler.assign_parameters_bounds( param_min, param_max )
    
    my_aap = aap.affine_advection_problem( my_parameter_handler )
    
    num_affine_components_A = 3
    num_affine_components_f = 3
    
    # defining the affine decomposition structure
    my_affine_decomposition = ad.AffineDecompositionHandler( )
    my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms
    
    # building the RB manager
    my_rb_manager = rm.RbManager( my_affine_decomposition, my_aap )
        
    my_rb_manager.import_basis_matrix( 'rb_structures/basis_affine_advection_' + str(rb_tol) + 'tol.txt' )
    
    basis = my_rb_manager.get_basis()
    my_affine_decomposition.import_rb_affine_matrices( 'rb_structures/rb_affine_components_affine_advection_' + str(rb_tol) + 'tol_A' )
    my_affine_decomposition.import_rb_affine_vectors(  'rb_structures/rb_affine_components_affine_advection_' + str(rb_tol) + 'tol_f' )
    norm_matrix = my_affine_decomposition.get_rb_affine_matrix(0)
    
    Yrb = Y.dot(basis)
    ntests = Yrb.shape[0]
    
    errors_rb = np.zeros( (ntests,1) )
    errors_pdenn = np.zeros( (ntests,1) )
    
    params1_m = np.zeros( (ntests,1) )
    params2_m = np.zeros( (ntests,1) )
    
    for i in runs_range:
        test = np.load( resdir + dirname + '/' + network_type + '/' + str(i) + '/test.npy' )
        for j in range(ntests):
            # my_rb_manager.solve_reduced_problem( test[j,-2:] )
            param0 = mu0_min + (mu0_max-mu0_min) * y_complete[j,-2] 
            param1 = mu1_min + (mu1_max-mu1_min) * y_complete[j,-1]
            my_rb_manager.solve_reduced_problem( np.array([param0,param1]) )
            Urb = my_rb_manager.M_un
            err = Urb - np.reshape(Yrb[j,:], Urb.shape )
            errscalar_rb = np.sqrt(err.transpose().dot(norm_matrix.dot(err)))
            normex = np.sqrt(Urb.transpose().dot(norm_matrix.dot(Urb)))
            errors_rb[j] = errscalar_rb/normex
            
            params1_m[j] = params1_m[j] + test[j,-2]/runs
            params2_m[j] = params2_m[j] + test[j,-1]/runs

            param0 = mu0_min + (mu0_max-mu0_min) * test[j,-2] 
            param1 = mu1_min + (mu1_max-mu1_min) * test[j,-1]
            my_rb_manager.solve_reduced_problem( np.array([param0,param1]) )
            Urb = my_rb_manager.M_un
            err = Urb - np.reshape(Yrb[j,:], Urb.shape )
            errscalar_pdenn = np.sqrt(err.transpose().dot(norm_matrix.dot(err)))/normex
            errors_pdenn[j] = errors_pdenn[j] + errscalar_pdenn/runs
            
#    cdict = {'red': ((0.0, 0.0, 0.0), \
#                    (0.5, 1.0, 0.7), \
#                    (1.0, 1.0, 1.0)), \
#         'green': ((0.0, 0.0, 0.0), \
#                   (0.5, 1.0, 0.0), \
#                   (1.0, 1.0, 1.0)), \
#         'blue': ((0.0, 0.0, 0.0), \
#                  (0.5, 1.0, 0.0), \
#                  (1.0, 0.5, 1.0))}
#    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    im1 = ax1.scatter(np.reshape(y_complete[:,-2],params1_m.shape),np.reshape(y_complete[:,-1],params2_m.shape), edgecolors=[0,0,0], c=[1,1,1])        
    im1 = ax1.scatter(params1_m, params2_m, c=np.log10(errors_pdenn))
    f.colorbar(im1,ax=ax1)
    im2 = ax2.scatter(np.reshape(y_complete[:,-2],params1_m.shape),np.reshape(y_complete[:,-1],params2_m.shape), c=np.log10(errors_rb))
    f.colorbar(im2,ax=ax2)
    print( "Average H1 error (pde): \t" + str( np.mean(errors_pdenn) ) )
    print( "Average H1 error (rb) : \t" + str( np.mean(errors_rb) ) )

    ax1.set_ylim( [0,1] )
    ax1.set_xlim( [0,1] )
    ax2.set_ylim( [0,1] )
    ax2.set_xlim( [0,1] )
    
    
inr = 20
outr = 20
tol = 1e-5
ns = 20000
ep =  500
y = np.load( 'input_arrays/y_test_20000samples_1e-07tol.npy' )
ycomplete = np.load( 'input_arrays/y_complete.npy' )
dirname = generate_dir_name( inr, outr, tol, ns, ep )
if dirname:
    plot_histories( 'pde', 'loss', dirname )
    plot_histories( 'pde', 'mean_absolute_error', dirname )
    plot_histories_loss_mae( 'pde', dirname )
    plot_prediction_parameters( 'pde', y, dirname )
    # plot_prediction_parameters( 'tf', y, dirname )
    # plot_prediction_parameters( 'tf_mu', y, dirname )
    # comparison_prediction_parameters( y, dirname )
    # scatterplot_of_errors( ycomplete, dirname )
    compute_H1_error( ycomplete, tol, 'pde', dirname )
    # print_rt_information( dirname )
    





