print("Import modules.")

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import math
import pytraj as pt
import matplotlib as mp
mp.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import pdist, squareform

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# vis
import altair as alt
#alt.renderers.enable("default")
alt.renderers.enable('notebook')
alt.data_transformers.disable_max_rows()
from plotly import graph_objs as go
import plotly.figure_factory as ff

#%matplotlib inline

print("Define data variables.")
input_trajectory = '/run/user/1000/gvfs/sftp:host=gullveig.intra.ecu.edu,user=sanchezw/data/people/sanchezw/work/GPU/proteins/kim/6_analysis/image/data/CA_full_traj_aligned_1_256.nc'
fixed_topology = '/run/user/1000/gvfs/sftp:host=gullveig.intra.ecu.edu,user=sanchezw/data/people/sanchezw/work/GPU/proteins/kim/6_analysis/image/data/CA_full_traj_aligned_1_256.top'

print("Define functions.")
def names():
	atom=traj.topology.residues
	list1=[]
	for ii in atom:
		some=str(ii).split()[0].strip('<,0123456789')
		list1.append(some)
	return list1

def id_info(res_num):

	if res_num <=345 and res_num >= 1:
		sub_name = 'A1I'
		seq= res_num
	elif res_num >=346 and res_num <= 450:
                sub_name = 'SI'
                seq= res_num - 345 +1
	elif res_num >=451 and res_num <= 555:
                sub_name = 'SII'
                seq= res_num - 450 +1
	elif res_num >=556 and res_num <= 900:
                sub_name = 'A1II'
                seq= res_num - 555 +1

	if (res_num >=1 and res_num <=39) or (res_num >=556 and res_num <=594):
		domain_name ='N'
	elif (res_num >=40 and res_num <=112) or (res_num >=595 and res_num <=667):
		domain_name ='RI'
	elif (res_num >=113 and res_num <=184) or (res_num >=668 and res_num <=739):
		domain_name ='RII'
	elif (res_num >=185 and res_num <=261) or (res_num >=740 and res_num <=816):
		domain_name ='RIII'
	elif (res_num >=262 and res_num <=345) or (res_num >=817 and res_num <=900):
		domain_name ='RIV'
	elif (res_num >=346 and res_num <= 400) or (res_num >=451 and res_num <=500):
		domain_name ='EFI'
	elif (res_num >=401 and res_num <=450) or (res_num >= 501 and res_num <=555):
		domain_name ='EFII'
	return [sub_name, domain_name, str(seq)]

'''
Purpose:
Arguments: (1) 2d array where each entry is a position vector.
           (2) 2d array where each entry is a position vector.
NOTES:     v1 and v2 must be the same length.
'''
def n_n_corr(v1, v2):
    a1=np.average(v1)
    a2=np.average(v2)
    answer = (np.dot(a1, a2)- np.linalg.norm(np.cross(a1, a2))) \
                 /                                  \
                 ((np.dot(a1, a1) - np.linalg.norm(np.cross(a1, a1))) * (np.dot(a2, a2) - np.linalg.norm(np.cross(a2, a2))))**0.5
    return answer

print("Import trajectory information.")
traj = pt.load(input_trajectory, fixed_topology)

t0 = traj.xyz

df = pd.DataFrame()
dict1 = dict()
#for frame in t0:

list3=range(1,900,1)
list2=[]
for i in list3:
	info = id_info(i)
	info= info[0] + '.' + info[1] + '.' +names()[i-1]+ info[2]
	list2.append(info)
#trajectory = pd.DataFrame(columns = list2)

ii = 1
for frame in t0:
	frame = [np.array([coord]) for coord in frame]
	if ii == 1:
		trajectory = np.array(frame)
	else:
		trajectory = np.concatenate((trajectory, frame), axis = 1)
	ii += 1

trajectory = list(trajectory)
trajectory = [list(ii) for ii in trajectory]
traj = pd.DataFrame.from_dict(dict(zip(list2, trajectory)))
#print(traj.iloc[:,2])
#print(np.corrcoef(traj.iloc[:,2], traj.iloc[:,3]))


#n_n_corr(traj.iloc[:,0], traj.iloc[:,1])
#print(traj.corr(method="nncorr"))
print("Calculate correlation matrix.")
cor_data = dict()
ii = 0
steps = len(list(traj.columns))
size = int(steps / 10)
print("0% ", end = "")
for col1 in traj.columns:
    cor_data[col1] = []
    ii += 1
    if ii % size == 0:
        print(str(int(ii / steps) * 100) + " %", end = "")
    for col2 in traj.columns:
        col1_data = np.array(traj[col1])
        col2_data = np.array(traj[col2])
        cor = n_n_corr(col1_data, col2_data)
        cor_data[col1].append(cor)
        
corr = pd.DataFrame.from_dict(cor_data, orient = 'index', 
                       columns = list(traj.columns))
corr = corr.reindex(list2, columns = list2)
corr

# create dictionary with column name as key and cluster as value
clusters = dict()
for col in traj.columns:
    clusters[col] = col.split('.')[0]

def visualize_feature_correlation(data, correlation, chart_type, target_name = None, cluster_scope = None, user_defined_clusters = None):
    df = data.copy()
    corr = correlation
    if target_name is not None:
        target_corr = np.abs(corr.rename({target_name:"target_corr"}, axis=1)["target_corr"])
        df = df.drop(target_name, axis=1)
        corr = corr.drop(target_name, axis=1).drop(target_name, axis=0)
    default_feature_order = sorted(list(df.columns))
    corr_condensed = hc.distance.squareform(1 - np.abs(correlation) ) # convert to condensed
    z = hc.linkage(corr_condensed, method='average');
    feature_order = hc.dendrogram(z, labels=df.columns, no_plot=True)["ivl"];
    if chart_type == "clusters":

        sidebar_width = 200
        sidebar_component_height = 75

        #compute PCA and store as X,Y coordinates for each feature
        pca = PCA(n_components = 2)
        pca.fit(np.abs(corr))
        names = list(df.columns)
        coords = pca.transform(np.abs(corr)).tolist()
        pca_coords = dict(zip(names, coords))
        pca_coords = pd.DataFrame.from_dict(pca_coords, orient = "index")
        pca_coords = pca_coords.reset_index()
        pca_coords = pca_coords.rename({0: "X", 1: "Y", "index": "feature"}, axis = 1)
        print(pca_coords)

        if user_defined_clusters:

            num_labels = np.unique(user_defined_clusters.values()).shape[0]
            silhouette_scores = [
                {
                    "cluster_num": num_labels,
                    "silhouette_score": 1,
                    "feature": col,
                    "cluster": user_defined_clusters[col]
                }
                for col in df.columns
            ]
        else:

            #get feature clusters via another method
            scaler = StandardScaler()
            feature_distances = squareform(pdist(scaler.fit_transform(df).T, "euclidean"))
            silhouette_scores = []
            if cluster_scope is None:
                cluster_range = range(3,df.shape[1])
            elif isinstance(cluster_scope, int):
                cluster_range = range(cluster_scope, cluster_scope + 1)
            else:
                cluster_range = cluster_scope #range object
            for n_cluster in cluster_range:
                corr_clusters = FeatureAgglomeration(n_clusters = n_cluster, affinity = "precomputed", linkage = "average").fit(feature_distances)
                silhouette_scores = silhouette_scores\
                + [
                    {
                        "cluster_num": n_cluster,
                        "silhouette_score": silhouette_score(feature_distances, corr_clusters.labels_, metric = "precomputed"),
                        "feature": list(df.columns)[i],
                        "cluster": label
                    }
                    for i, label in enumerate(corr_clusters.labels_)
                ]

        cluster_label_df = pd.DataFrame(silhouette_scores)
        cluster_label_df["cluster_size"] = cluster_label_df.groupby(["cluster_num", "cluster"])["feature"].transform("count")
        cluster_label_df["key"] = cluster_label_df["cluster_num"].astype(str).str.cat(cluster_label_df["feature"].astype(str), sep=":")

        cluster_label_df["cluster"] = cluster_label_df.groupby(["cluster_num","cluster"])["feature"].transform("first")

        default_cluster_num = cluster_label_df.groupby("cluster_num")["silhouette_score"].max().idxmax()

        # set correlation with target, if using, which determines circle size
        if target_name is not None:
            pca_coords = pca_coords.join(
                target_corr
            ).reset_index()
        else:
            pca_coords = pca_coords.reset_index()
            pca_coords["target_corr"] = 1

        # get dataset for lines between features (if they have higher correlation than corr_threshold)
        corr_lines = corr.reset_index(drop=False).rename({"index":"feature"}, axis=1)\
            .melt(id_vars = ["feature"], var_name = "feature_2", value_name = "corr")\
            .query("feature > feature_2")

        corr_lines["corr_abs"] = np.abs(corr_lines["corr"])
        corr_selector_data = corr_lines.copy()
        corr_selector_data["corr_abs"] = np.floor((corr_selector_data["corr_abs"]*10))/10
        corr_selector_data = corr_selector_data.groupby("corr_abs").size().reset_index().rename({0:"Count"}, axis = 1)
        corr_lines_1 = pd.merge(
            corr_lines,
            pca_coords.loc[:,["feature", "X", "Y"]],
            on = "feature"
        )
        corr_lines_2 = pd.merge(
            corr_lines,
            pca_coords.set_index("feature").loc[:,["X", "Y"]],
            left_on = "feature_2", right_index = True
        )
        corr_lines = corr_lines_1.append(corr_lines_2)
        corr_lines["key"] = corr_lines["feature"] + corr_lines["feature_2"]

        corr_line_selector = alt.selection_single(fields = ["corr_abs"], init = {"corr_abs":0.7})
        cluster_num_selector = alt.selection_single(fields = ["cluster_num"], init = {"cluster_num":default_cluster_num})
        cluster_selection = alt.selection_single(fields=["cluster"])

        base = alt.layer().encode(
            x = alt.X("X", axis=None),
            y = alt.Y("Y", axis=None),
            color = alt.condition(
                cluster_selection,
                alt.Color("cluster:N", legend = None),
                alt.value("lightgray")
            )
        )

        base += alt.Chart(pca_coords).mark_circle().encode(
            size = alt.Size("target_corr:Q", scale=alt.Scale(domain = [0,1]), legend=None)
        ).transform_calculate(
            key = cluster_num_selector.cluster_num + ":" + alt.datum.feature
        ).transform_lookup(
            lookup='key',
            from_=alt.LookupData(data=cluster_label_df, key='key',
                                 fields=['cluster_size', 'cluster'])
        ).add_selection(
            cluster_num_selector
        )

        base += alt.Chart(pca_coords).mark_text(dx=20, dy = 10).encode(
            text = "feature",
        ).transform_calculate(
            key = cluster_num_selector.cluster_num + ":" + alt.datum.feature
        ).transform_lookup(
            lookup='key',
            from_=alt.LookupData(data=cluster_label_df, key='key',
                                 fields=['cluster_size', 'cluster'])
        )

        base += alt.Chart(corr_lines).mark_line().encode(
            detail = "key",
            strokeWidth = alt.StrokeWidth("corr_abs", scale = alt.Scale(domain = [0,1], range = [.3,3]))
        ).transform_filter(
            alt.datum.corr_abs >= corr_line_selector.corr_abs
        ).transform_calculate(
            key = cluster_num_selector.cluster_num + ":" + alt.datum.feature
        ).transform_lookup(
            lookup='key',
            from_=alt.LookupData(data=cluster_label_df, key='key',
                                 fields=['cluster_size', 'cluster'])
        )

        base = base.properties(
            width = 800,
            height = 500,
            title = "Feature Space Diagram"
        ).interactive()

        num_cluster_picker = alt.Chart(cluster_label_df).mark_bar().encode(
            y = alt.Y("silhouette_score", title = "Silhouette Score"),
            x = "cluster_num:O",
            color = alt.condition(
                cluster_num_selector,
                alt.value("lightblue"),
                alt.value("lightgray")
            )
        ).add_selection(
            cluster_num_selector
        ).properties(
            width = sidebar_width,
            height = sidebar_component_height,
            title = "Select the Number of Clusters"
        )

        corr_threshold_picker = alt.Chart(corr_selector_data).mark_bar().encode(
            x = "corr_abs:O",
            y = alt.Y("Count", axis = alt.Axis(labelAngle = 0, title = "Feature Pairs")),
            color = alt.condition(
                alt.datum.corr_abs >= corr_line_selector.corr_abs,
                alt.value("lightblue"),
                alt.value("lightgray")
            )
        ).add_selection(
            corr_line_selector
        ).properties(
            width = sidebar_width,
            height = sidebar_component_height,
            title = "Select Correlation Threshold to Show Lines"
        )

        cluster_bar_chart = alt.Chart(cluster_label_df).mark_bar(size=5).encode(
            y = alt.Y(
                "cluster:N",
                sort = alt.EncodingSortField(field = "cluster_size", order="descending"),
                title = None #  "Clusters"
            ),
            x = "cluster_size",
            color = alt.Color("cluster:N", legend=None),
        ).add_selection(
            cluster_selection
        ).transform_filter(
            (alt.datum.cluster_num >= cluster_num_selector.cluster_num) & (alt.datum.cluster_num <= cluster_num_selector.cluster_num)
        ).properties(
            width = sidebar_width,
            height = 200,
            title = "Cluster Sizes. Click to Highlight"
        )

        return (base) | (num_cluster_picker & corr_threshold_picker & cluster_bar_chart)

print("Visualize.")
visualize_feature_correlation(traj, corr, "clusters", user_defined_clusters = clusters)
