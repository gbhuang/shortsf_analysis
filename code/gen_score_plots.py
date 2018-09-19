import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors
import matplotlib.mlab
import scipy.stats
import seaborn as sns
import textwrap

# preliminaries
ratings  = pd.read_csv('csv/ratings.csv')
attr     = pd.read_csv('csv/attr.csv')
notes    = pd.read_csv('csv/notes.csv')

for ii in range(np.size(notes,0)):
    if not isinstance(notes.notes[ii], str):
        notes.notes[ii] = '[summary needed]'
    else:
        notes.notes[ii] = '<br />'.join(textwrap.wrap(
            notes.notes[ii], 50))

new_columns = []
for cc in ratings.columns:
    new_columns.append(cc.replace('\n',''))
ratings.columns = new_columns

n_raters = 13;
ratings_shape = ratings.shape
all_ratings   = ratings.iloc[
    :, [ratings_shape[1]-2,]+list(range(1,n_raters+1))]
ratings  = ratings.iloc[:,1:n_raters+1]


# rater distributions
plt.figure(figsize=(10,6))
ax = sns.violinplot(ratings,orient='v',scale='width',cut=0,gridsize=50)
ax.set_ylim(1,5)
plt.xticks(np.arange(13),ratings.columns,rotation=45)
plt.ylabel('score')
plt.savefig('images/rater_distribution.png')
plt.close()


# rater correlations
corr = np.zeros( (n_raters+1,n_raters+1) )
n_smooth = 10
n_iters  = 500
for ii in range(n_iters):
    corr_smoothing = pd.DataFrame(
        np.random.randn(n_smooth,n_raters+1)+3.,
        columns=all_ratings.columns)
    ratings_smoothed = pd.concat([all_ratings, corr_smoothing])
    cc = ratings_smoothed.corr()
    corr += cc.values
corr /= n_iters
np.fill_diagonal(corr, np.nan)

c2 = all_ratings.corr(min_periods=10)
idx = np.isnan(c2)
corr[idx.values] = np.nan

plt.figure(figsize=(8,8))
sns.heatmap(corr,annot=True,fmt='.02f',linewidths=1, cbar=False)
plt.xticks(0.5+np.arange(n_raters+1),all_ratings.columns,rotation=45)
plt.yticks(0.5+np.arange(n_raters,-1,-1),all_ratings.columns,rotation=0)
plt.savefig('images/rater_correlations.png')


# author_means.png
#   box plot for each author with multiple stories,
#   and box plots for all multiple, all single
authors = {}
for aa in attr.itertuples():
    if aa.author not in authors:
        authors[aa.author] = []
    authors[aa.author].append(aa.b_mean)
all_scores = [ [], [] ]
labels = [ 'all single', 'all multiple' ]
for aa, mm in sorted(authors.items(),
                     key=lambda key:np.mean(key[1])):
    if len(mm)==1:
        all_scores[0] += mm
    else:
        all_scores[1] += mm
        all_scores.append(mm)
        labels.append('%s (%d)' % (aa, len(mm)))
plt.figure(figsize=(14,14))
plt.boxplot(all_scores,vert=False,labels=labels)
plt.xlabel('story mean')
plt.gca().xaxis.grid(True)
plt.savefig('images/author_means.png')
plt.close()


# chooser_self_bias.png
#   box plot for person's ratings, when chooser and not
mean_rating = attr.b_mean
deviation   = ratings.sub(mean_rating, axis=0)
dev_dist    = []
labels      = []
sb_info     = []

for ii in range(0,n_raters):
    vv = deviation.values[:,ii]
    select = np.array(
        attr.chooser==ratings.columns[ii],
        dtype=bool)
    vv_out = vv[~select]
    vv_in  = vv[ select]
    vv_out = vv_out[~np.isnan(vv_out)]
    vv_in  = vv_in[ ~np.isnan(vv_in )]

    ss   = scipy.stats.ttest_ind(vv_out,vv_in)
    sgnf = ''
    if(ss.pvalue/2 < 0.05):
        sgnf = '*'
    if(ss.pvalue/2 < 0.01):
        sgnf = '**'

    sb_info.append( [-ss.pvalue,
                     [vv_out,vv_in,ratings.columns[ii],sgnf]] )

for ii in sorted(sb_info):
    if np.isnan(ii[0]):
        continue
    dev_dist.append(ii[1][0])
    dev_dist.append(ii[1][1])
    labels.append(
        '%s, not choose' % ii[1][2])
    labels.append(
        '%s%s, choose' % (ii[1][2], ii[1][3]))

plt.figure(figsize=(14,5))
plt.boxplot(dev_dist,vert=False,labels=labels)
plt.xlabel('deviation from story mean')
plt.gca().xaxis.grid(True)
plt.savefig('images/chooser_self_bias.png')
plt.close()


# chooser_stds.png
#   box plot for chooser's standard deviations
std_all = attr['b_std'];
std_dist = []
for cc in attr.chooser.unique():
    select = np.array(attr.chooser==cc, dtype=bool)
    ss = std_all[select]
    std_dist.append([ss, cc])
std_dist.sort(key=lambda key:np.median(key[0]))

std_d = []
std_l = []
for ll in std_dist:
    std_d.append(ll[0])
    std_l.append(ll[1])
plt.figure(figsize=(14,5))
plt.boxplot(std_d,vert=False,labels=std_l)
plt.xlabel('story std')
plt.gca().xaxis.grid(True)
plt.savefig('images/chooser_stds.png')
plt.close()


# stories_plot.png
#   bubble plot of means,stds
plt.figure(figsize=(20,14))
n_stories = np.size(attr,0)
r_mn  = attr.b_mean + np.random.rand(n_stories)/1000;
r_std = attr.b_std  + np.random.rand(n_stories)/1000;
s_mn  = []
s_std = []
s_sz  = []
s_cl  = []
plotly_hover = []
choosers = attr.chooser.unique()
for ii in range(0,n_stories):
    s_mn.append(np.sum(r_mn[ii]  >r_mn  ) / n_stories )
    s_std.append(np.sum(r_std[ii]>r_std ) / n_stories )
    s_sz.append(attr.M[ii]*attr.M[ii]*40)
    c_idx = np.flatnonzero(choosers==
                           attr.chooser[ii]);
    s_cl.append(c_idx[0])
    if c_idx[0]==0: # novel
        s_sz[-1] *= 1.4

    plotly_hover.append(
        '%s<br />author: %s<br />"%s"<br /><br />chooser: %s<br />mean: %g<br />std: %g' % (
            attr.title[ii], attr.author[ii], notes.notes[ii],
            choosers[c_idx[0]],
            attr.b_mean[ii], attr.b_std[ii]) )

xx = np.array([s_mn, s_std])
dd_min   = 0.02
dd_shift = 0.001
n_iter = 0
while(True):
    n_shift = 0
    dd = np.sqrt(np.sum(np.square(xx.reshape( (2,n_stories,1) ) -
                                  xx.reshape( (2,1,n_stories) )),
                        axis=0))
    for ii in range(0,n_stories):
        for jj in range(0,n_stories):
            if ii==jj or dd[ii,jj] >= dd_min:
                continue
            if xx[1,ii] > 0.45 and xx[1,ii] < xx[1,jj]:
                xx[1,ii] = xx[1,ii] - dd_shift
                n_shift = n_shift + 1
                #print('%d up' % ii)
            if xx[1,ii] < 0.55 and xx[1,ii] > xx[1,jj]:
                xx[1,ii] = xx[1,ii] + dd_shift
                n_shift = n_shift + 1
                #print('%d down' % ii)
    n_iter = n_iter + 1
    print(n_shift)
    if n_shift==0 or n_iter>200:
        break

cc = sns.color_palette('Set1',np.max(s_cl))
cc.insert(0, (0, 0, 0))
#cc = sns.husl_palette(np.max(s_cl)+1, l=.5, s=.9)
cm = matplotlib.colors.ListedColormap(cc)

s_mn  = xx[0,:]
s_std = xx[1,:]
for ii in range(0,n_stories):
    plt.text(s_mn[ii], s_std[ii], attr.title[ii], size=9,
             horizontalalignment='center',
             verticalalignment='center',
             path_effects=[pe.withStroke(linewidth=2,
                                         foreground='w')])
sct = plt.scatter(s_mn, s_std, c=s_cl, s=s_sz,
                  cmap=cm,  # plt.get_cmap('Paired'),
                  linewidths=2, edgecolor='w')
sct.set_alpha(0.8)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('story mean rank')
plt.ylabel('story std rank')
plt.tight_layout()
plt.savefig('images/stories_plot.png')
plt.close()


cc_list = []
for ii in s_cl:
    cc_int = np.floor(np.asarray(cc[ii])*255).astype(int)
    cc_list.append( 'rgb(%d,%d,%d)' %
                    (cc_int[0],cc_int[1],cc_int[2]) )

layout = go.Layout(
    hovermode = 'closest',
    xaxis=dict(title='story mean rank'),
    yaxis=dict(title='story std rank'))
plotly.offline.plot( {'data': [
    go.Scatter(x=s_mn, y=s_std,
               mode='markers',
               text=plotly_hover,
               hoverinfo='text',
               textfont=dict(
                   size=6,
                   color='#cccccc'
                   ),
               marker=dict(
                   color=cc_list,
                   size=np.asarray(s_sz),
                   sizemode='area',
                   sizeref=2.*max(s_sz)/(50.**2),
                   sizemin=4
               ))],
                      'layout': layout},
                     filename='images/stories_plot.html')


# histograms classics versus contemporary
if False:
    plt.figure(figsize=(6,6))
    plt.hist([attr.b_mean[attr.year<2010], attr.b_mean[attr.year>=2010]],
             np.linspace(0.75,4.75,9), alpha=0.5,
             weights=[np.ones(np.sum(attr.year<2010))/np.sum(attr.year<2010),
                      np.ones(np.sum(attr.year>=2010))/np.sum(attr.year>=2010)],
             label=['before 2010', '2010 and after'])
    plt.tight_layout()

    ratings_sub = ratings.iloc[:,[0,1,3,5,7]]
    full_idx = (np.sum(np.isnan(ratings.iloc[:,[0,1,3,5,7]]),axis=1)==0).nonzero()[0]
    ratings_full = ratings_sub.iloc[full_idx,:]
    rf_mn0 = np.mean(ratings_full,axis=0).values.reshape(1,5)
    rf_std = np.std(ratings_full,axis=0).values.reshape(1,5)
    rf_norm0 = (ratings_full.values - rf_mn0)/rf_std
    rf_mn1 = np.mean(rf_norm0, axis=1).reshape(-1,1)
    rf_norm1 = rf_norm0 - rf_mn1

    results = mamtplotlib.mlab.PCA(rf_norm1, standardize=False)

    cc = sns.color_palette("RdBu_r", 100)

    rf_mn1_sc = (99*(rf_mn1 - np.min(rf_mn1)) /
                 (np.max(rf_mn1)-np.min(rf_mn1))).astype('int')
    cc_list = []
    for ii in rf_mn1_sc:
        cc_int = np.floor(np.asarray(cc[ii[0]])*255).astype(int)
        cc_list.append( 'rgb(%d,%d,%d)' %
                        (cc_int[0],cc_int[1],cc_int[2]) )

    layout = go.Layout(
        hovermode = 'closest',
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2'))
    plotly.offline.plot( {'data': [
        go.Scatter(x=results.Y[:,0], y=results.Y[:,1],
                   mode='markers',
                   text=[plotly_hover[ii] for ii in full_idx],
                   hoverinfo='text',
                   textfont=dict(
                       size=6,
                       color='#cccccc'
                       ),
                   marker=dict(
                       color=cc_list
                   ))],
                          'layout': layout},
                         filename='images/tmp_pca_plot.html')
