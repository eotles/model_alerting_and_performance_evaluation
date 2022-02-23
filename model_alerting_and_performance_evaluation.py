import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

def compute_alerts_and_performance(df, taus=np.linspace(0,1,25)):
    #calculate sensitivity at the population level
    #calculate alerts at the observer level
    
    res = []
    for tau in taus:
        _df = df.copy()
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


    plt.rcParams["figure.figsize"] = (15,15)
    taus = list(ap_res['tau'])
    extent = [taus[0], taus[-1], alpha[-1,0], alpha[0,0]]

    plt.imshow(v, extent=extent, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('tau')
    plt.ylabel('alpha\nupweight {} {} upweight {}'.format(c2, ' '*40, c1))
    plt.title('alpha*{} + (1-alpha)*{}\nin w.r.t alpha & tau'.format(c1, c2))
    plt.show()


