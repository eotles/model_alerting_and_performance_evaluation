import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


plt.rcParams["figure.figsize"] = (15,15)

def compute_alerts_and_performance(df, taus=np.linspace(0,1,25), return_oc_res=False, tau_on_percentile=False):
    """
    From a dataframe of model risk estimates returns the observer level alerts and performance of various decision thresholds (taus).
    
    ***

    Parameters
    ----------
    df : pandas.DataFrame
        Given dataframe of risk estimates must have the following columns: ***
       
    taus: np.array, optional
        The decision thresholds being considered
    
    return_oc_res: Boolean, optional
        If the observed alert counts should be returns
        
    tau_on_percentile: Boolean, optional
        If the taus are evaluated on the percentiles of the risk estimates

    Returns
    -------
    ap_res : returns summaries of the observer level number of alerts and performance for each tau
    
    oc_res : returns the actual number of alerts obvserved by each observer for a given tau
    """
    #calculate sensitivity at the population level
    #calculate alerts at the observer level

    observers = df['observer'].unique()

    res = []
    oc_res = []
    for tau in taus:
        _df = df.copy()
        if tau_on_percentile:
            _df['p'] = _df['p'].rank(pct=True)
        _df['y_hat'] = 1*(_df['p']>=tau)



        #population performance (confusion matrix (cm))
        _ = _df.groupby('eID').max()
        cm = {}
        tn, fp, fn, tp  = metrics.confusion_matrix(_['y'], _['y_hat']).ravel()
        cm['tn'] = tn
        cm['fp'] = fp
        cm['fn'] = fn
        cm['tp'] = tp

        #observer alerts
        _ = _df[_df['y_hat']==1].groupby('eID').first()
        alert_counts = _['observer'].value_counts()
        oc = alert_counts.to_dict()
        oc_v = oc.values()
        oa = {'oa_max': alert_counts.max(), 'oa_min': alert_counts.min(), 
              'oa_mean': alert_counts.mean(), 'oa_med': alert_counts.median(),
              'oa_sum': alert_counts.sum(), 
             }

        _oc_res = [{'observer': o, 'tau': tau, 'n_alerts': oc.get(o, 0)} for o in observers]
        oc_res+=_oc_res

        #save tau res
        _res = {'tau': tau}
        _res.update(cm)
        _res.update(oa)
        res.append(_res)


    ap_res = pd.DataFrame(res)
    #calculate performance based on cm
    ap_res['sens'] = ap_res['tp'] / (ap_res['tp'] + ap_res['fn'])
    ap_res['spec'] = ap_res['tn'] / (ap_res['tn'] + ap_res['fp'])
    ap_res['ppv'] = ap_res['tp'] / (ap_res['tp'] + ap_res['fp'])
    ap_res['npv'] = ap_res['tn'] / (ap_res['tn'] + ap_res['fn'])

    ne = _df['eID'].nunique()
    ap_res['proportion_unalerted'] = (ne-ap_res['oa_sum'])/ne

    oc_res = pd.DataFrame(oc_res)
    
    if return_oc_res:
        return(ap_res, oc_res)
    else:
        return(ap_res)


def plot_alerts_and_performance(ap_res, performance_measures=['sens', 'spec']):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    for m in performance_measures:
        ax1.plot(ap_res['tau'], ap_res[m], label=m)


    ax2.plot(ap_res['tau'], ap_res['oa_med'], ':', color='r', label='# alerts per unit')
    ax2.fill_between(ap_res['tau'], ap_res['oa_max'], ap_res['oa_min'], color='r', alpha=0.5,)

    ax1.set_xlabel('Tau')
    ax1.set_ylabel('Performance')
    ax2.set_ylabel('# Observed Alerts ')

    ax1.legend(loc='lower left')
    ax2.legend(loc='upper right')

    plt.show()
    
    
def plot_trade_off(ap_res,
                   alpha=np.expand_dims(np.linspace(1,0, 25), axis=1),
                   cmap = plt.cm.get_cmap('viridis'),
                   c1 = 'sens' , c2='proportion_unalerted' ):
    v1 = np.expand_dims(ap_res[c1].values, axis=0)
    v2 = np.expand_dims(ap_res[c2].values, axis=0)
    v = alpha*v1 + (1-alpha)*v2
    
    taus = list(ap_res['tau'])
    extent = [taus[0], taus[-1], alpha[-1,0], alpha[0,0]]

    plt.imshow(v, extent=extent, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('tau')
    plt.ylabel('alpha\nupweight {} {} upweight {}'.format(c2, ' '*40, c1))
    plt.title('alpha*{} + (1-alpha)*{}\nin w.r.t alpha & tau'.format(c1, c2))
    plt.show()
    
    #new
    _X,_Y = np.meshgrid(taus,  alpha)

    fig, ax = plt.subplots()
    CS = ax.contour(_X, _Y, v)
    ax.clabel(CS, CS.levels, inline=True, fontsize=10)
    fig.colorbar(CS)
    ax.set_xlabel('tau')
    ax.set_ylabel('alpha*{} + (1-alpha)*{}\nin w.r.t alpha & tau'.format(c1, c2))
    plt.show()

    
def plot_gain(ap_res,
              alpha=np.expand_dims(np.linspace(0.1,0.3, 5), axis=1), 
              kappa = 10000,
              beta = 20):
    
    #alpha: effectiveness
    #kappa: cost of CDI
    #beta: intervention cost
    
    n_intervene = np.expand_dims((ap_res['tp']+ap_res['fp']).values, axis=0)
    n_intervene_pos = np.expand_dims((ap_res['tp']).values, axis=0)
    v = alpha*kappa*n_intervene_pos - beta*n_intervene

    for i,_a in enumerate(alpha):
        plt.plot(list(ap_res['tau']), v[i], label='alpha={}'.format(_a[0]))

    plt.xlabel('tau')
    plt.ylabel('Gain')
    plt.legend()
    plt.show()