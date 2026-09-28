import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config("ClusterLab", "🔬", layout="wide")
st.markdown("""<style>.stApp{background:#f6f8fb}.block-container{max-width:1450px;padding-top:2rem}[data-testid='stSidebar']{background:#fff}.hero{background:linear-gradient(120deg,#102a43,#134e4a);border-radius:22px;padding:30px;margin-bottom:22px}.hero h1{color:#fff!important;margin:8px 0}.hero p{color:#c5d9e5;margin:0}.eyebrow{color:#5eead4;letter-spacing:3px;font-size:12px;font-weight:700}</style>""",unsafe_allow_html=True)
COLORS=["#14b8a6","#8b5cf6","#f59e0b","#f43f5e","#3b82f6","#ec4899"]

@st.cache_data
def demo():
    ds=load_breast_cancer(as_frame=True);df=ds.data.copy();df.insert(0,"diagnosis",ds.target.map({0:"M",1:"B"}));return df

def show(fig,h=420):
    fig.update_layout(template="plotly_white",height=h,paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=20,r=20,t=50,b=25),legend_title_text="")
    fig.update_xaxes(gridcolor="#edf2f7");fig.update_yaxes(gridcolor="#edf2f7")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

st.markdown("""<div class='hero'><div class='eyebrow'>UNSUPERVISED LEARNING WORKSPACE</div><h1>Discover the structure in your data.</h1><p>Interactive PCA and K-Means exploration for the Wisconsin Breast Cancer dataset.</p></div>""",unsafe_allow_html=True)
with st.sidebar:
    st.title("🔬 ClusterLab");st.caption("PCA + K-Means Explorer");st.divider()
    source=st.radio("Dataset",["Built-in Wisconsin dataset","Upload CSV"])
    upload=st.file_uploader("Choose CSV",type="csv") if source=="Upload CSV" else None
if source=="Upload CSV":
    if upload is None: st.info("Upload a CSV from the sidebar to begin.");st.stop()
    try: data=pd.read_csv(io.BytesIO(upload.getvalue()))
    except Exception: st.error("Could not read this CSV.");st.stop()
else: data=demo()
data.columns=data.columns.astype(str).str.strip()
excluded={"id","diagnosis","target","label","class","cluster"}
available=[c for c in data if c.lower() not in excluded and not c.lower().startswith("unnamed:") and pd.api.types.is_numeric_dtype(data[c])]
with st.sidebar:
    st.divider();st.subheader("Analysis settings")
    features=st.multiselect("Measurement features",available,default=available)
    drop_dupes=st.checkbox("Remove duplicate feature rows")
    seed=int(st.number_input("Random seed",0,100000,42));n_init=st.selectbox("K-Means initializations",[10,20,30])
if len(features)<2: st.warning("Select at least two numeric features.");st.stop()
X=data[features].replace([np.inf,-np.inf],np.nan);keep=~X.isna().all(axis=1)
if drop_dupes: keep&=~X.duplicated()
clean=data.loc[keep].copy();X=X.loc[keep].copy();dropped=[c for c in X if X[c].nunique(dropna=True)<=1];X=X.drop(columns=dropped)
if len(X)<3 or X.shape[1]<2: st.error("Need 3 usable rows and 2 nonconstant features.");st.stop()
missing=int(X.isna().sum().sum());values=SimpleImputer(strategy="median").fit_transform(X);scaled=StandardScaler().fit_transform(values)
pca=PCA(n_components=2);Z=pca.fit_transform(scaled);variance=pca.explained_variance_ratio_;max_k=min(10,len(Z)-1,len(np.unique(Z,axis=0)))
if max_k<2: st.error("Need at least two distinct PCA points.");st.stop()
models={};rows=[];sample=np.random.default_rng(seed).choice(len(Z),min(2000,len(Z)),replace=False)
for k0 in range(1,max_k+1):
    m=KMeans(n_clusters=k0,random_state=seed,n_init=n_init);labels0=m.fit_predict(Z);sil=np.nan
    if k0>1 and len(np.unique(labels0[sample]))>1: sil=silhouette_score(Z[sample],labels0[sample])
    models[k0]=m;rows.append({"k":k0,"Inertia":m.inertia_,"Silhouette":sil})
metrics=pd.DataFrame(rows)
with st.sidebar: k=st.selectbox("Number of clusters (k)",range(2,max_k+1));st.caption("Educational exploration only — not medical diagnosis.")
model=models[k];labels=model.labels_;names=[f"Cluster {i+1}" for i in range(k)];palette=dict(zip(names,COLORS))
points=pd.DataFrame({"PC1":Z[:,0],"PC2":Z[:,1],"Cluster":[f"Cluster {x+1}" for x in labels],"Source row":clean.index+1})
diag=next((c for c in data if c.lower()=="diagnosis"),None)
if diag: points["Diagnosis"]=clean[diag].astype(str).str.upper().map({"B":"Benign","M":"Malignant"}).fillna("Unknown")
score=metrics.loc[metrics.k==k,"Silhouette"].iloc[0]
a,b,c,d=st.columns(4);a.metric("Samples analyzed",f"{len(clean):,}");b.metric("Features",len(X.columns));c.metric("PCA variance retained",f"{variance.sum():.1%}");d.metric("Silhouette","N/A" if pd.isna(score) else f"{score:.3f}")
overview,choose,insights,exports,guide=st.tabs(["Cluster overview","Choose k","Feature insights","Dataset & export","How it works"])
with overview:
    left,right=st.columns([2.2,1])
    with left:
        by=st.radio("Color points by",["Cluster"]+(["Diagnosis"] if diag else []),horizontal=True)
        cmap=palette if by=="Cluster" else {"Benign":"#14b8a6","Malignant":"#f43f5e","Unknown":"#94a3b8"}
        fig=px.scatter(points,x="PC1",y="PC2",color=by,opacity=.76,color_discrete_map=cmap,hover_data=["Source row","Cluster"],labels={"PC1":f"PC1 · {variance[0]:.1%} variance","PC2":f"PC2 · {variance[1]:.1%} variance"})
        if by=="Cluster": fig.add_trace(go.Scatter(x=model.cluster_centers_[:,0],y=model.cluster_centers_[:,1],mode="markers",name="Centroids",marker=dict(symbol="x",size=18,color="#172b4d")))
        show(fig,510)
    with right:
        counts=points.Cluster.value_counts().reindex(names,fill_value=0).rename_axis("Cluster").reset_index(name="Samples")
        show(px.pie(counts,names="Cluster",values="Samples",hole=.7,color="Cluster",color_discrete_map=palette),330);st.dataframe(counts,hide_index=True,use_container_width=True)
    if diag and (points.Diagnosis!="Unknown").any(): show(px.imshow(pd.crosstab(points.Cluster,points.Diagnosis),text_auto=True,aspect="auto",color_continuous_scale="Teal",title="Cluster versus known diagnosis"),300)
with choose:
    x,y=st.columns(2)
    with x: show(px.line(metrics,x="k",y="Inertia",markers=True,title="Elbow method",color_discrete_sequence=["#0d9488"]))
    with y: show(px.line(metrics.dropna(),x="k",y="Silhouette",markers=True,title="Silhouette analysis",color_discrete_sequence=["#8b5cf6"]))
    st.dataframe(metrics.round(4),hide_index=True,use_container_width=True)
with insights:
    profiles=pd.DataFrame(scaled,columns=X.columns).groupby(labels).mean();profiles.index=names;top=profiles.var(axis=0).nlargest(min(12,len(X.columns))).index
    show(px.imshow(profiles[top],text_auto=".1f",aspect="auto",color_continuous_scale="RdBu_r",color_continuous_midpoint=0,title="Cluster profiles · mean z-scores"),390)
    feature=st.selectbox("Inspect a feature",X.columns);frame=pd.DataFrame({feature:values[:,list(X.columns).index(feature)],"Cluster":points.Cluster})
    show(px.box(frame,x="Cluster",y=feature,color="Cluster",color_discrete_map=palette,points="outliers",title="Feature distribution by cluster"))
with exports:
    a,b,c,d=st.columns(4);a.metric("Original rows",len(data));b.metric("Rows removed",len(data)-len(clean));c.metric("Cells median-imputed",missing);d.metric("Constant features excluded",len(dropped))
    result=clean.copy();result["analysis_cluster"]=labels+1;result["analysis_PC1"]=Z[:,0];result["analysis_PC2"]=Z[:,1];st.dataframe(result,hide_index=True,use_container_width=True)
    l,r=st.columns(2);l.download_button("Download clustering results",result.to_csv(index=False).encode(),"clustering_results.csv","text/csv",use_container_width=True);r.download_button("Download evaluation metrics",metrics.to_csv(index=False).encode(),"cluster_evaluation.csv","text/csv",use_container_width=True)
with guide:
    st.markdown("""1. Select numeric measurements.  
2. Median-impute and standardize.  
3. Project onto PC1 and PC2.  
4. Fit K-Means on the PCA projection.  
5. Compare inertia and silhouette scores before choosing k.""")
    st.info(f"PC1 and PC2 retain {variance.sum():.1%} of total variance.")
st.divider();st.caption("ClusterLab · Wisconsin Breast Cancer · PCA + K-Means")
