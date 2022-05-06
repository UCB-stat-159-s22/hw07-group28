import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

def encode_age_category(x):
	category = {'55-59':57, '80 or older':80, '65-69':67,'75-79':77,'40-44':42,'70-74':72,'60-64':62,
				'50-54':52,'45-49':47,'18-24':21,'35-39':37,
				'30-34':32,'25-29':27}
	return category[x]

def make_numerical_tbl(df):
	 return df.describe()[1:][['BMI','PhysicalHealth','MentalHealth', 'Age', 'SleepTime']].T.style.background_gradient(cmap='Blues')

def plot_piechart(df):
	fig = make_subplots(rows=7, cols=2, subplot_titles=("HeartDisease", "Smoking",
                                    "AlcoholDrinking","Stroke",
                                    "DiffWalking", "Sex",
                                    'Race', 'Diabetic',
                                    'PhysicalActivity','GenHealth',
                                    'Asthma', 'KidneyDisease',
                                    'SkinCancer'),
						specs=[[{"type": "domain"}, {"type": "domain"}],[{"type": "domain"}, {"type": "domain"}],
							   [{"type": "domain"}, {"type": "domain"}],
							   [{"type": "domain"}, {"type": "domain"}],
							   [{"type": "domain"}, {"type": "domain"}],
							   [{"type": "domain"}, {"type": "domain"}],
							   [{"type": "domain"}, {"type": "domain"}]])

	colours = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']

	fig.add_trace(go.Pie(labels=np.array(df['HeartDisease'].value_counts().index),
						 values=[x for x in df['HeartDisease'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=1, col=1)

	fig.add_trace(go.Pie(labels=np.array(df['Smoking'].value_counts().index),
						 values=[x for x in df['Smoking'].value_counts()], hole=.35,
						 textinfo='label+percent', marker_colors=colours), row=1, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['AlcoholDrinking'].value_counts().index),
						 values=[x for x in df['AlcoholDrinking'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=2, col=1)

	fig.add_trace(go.Pie(labels=np.array(df['Stroke'].value_counts().index),
						 values=[x for x in df['Stroke'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=2, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['DiffWalking'].value_counts().index),
						 values=[x for x in df['DiffWalking'].value_counts()], hole=.35,
						 textinfo='label+percent', marker_colors=colours),row=3, col=1)

	fig.add_trace(go.Pie(labels=np.array(df['Sex'].value_counts().index),
						 values=[x for x in df['Sex'].value_counts()], hole=.35,
						 textinfo='label+percent', marker_colors=colours),row=3, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['Race'].value_counts().index),
						 values=[x for x in df['Race'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=4, col=1)

	fig.add_trace(go.Pie(labels=np.array(df['PhysicalActivity'].value_counts().index),
						 values=[x for x in df['PhysicalActivity'].value_counts()], hole=.35,
						 textinfo='label+percent', marker_colors=colours),row=4, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['Diabetic'].value_counts().index),
						 values=[x for x in df['Diabetic'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=5, col=1)


	fig.add_trace(go.Pie(labels=np.array(df['GenHealth'].value_counts().index),
						 values=[x for x in df['GenHealth'].value_counts()], hole=.35,
						 textinfo='label+percent', marker_colors=colours),row=5, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['Asthma'].value_counts().index),
						 values=[x for x in df['Asthma'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=6, col=1)

	fig.add_trace(go.Pie(labels=np.array(df['KidneyDisease'].value_counts().index),
						 values=[x for x in df['KidneyDisease'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=6, col=2)

	fig.add_trace(go.Pie(labels=np.array(df['SkinCancer'].value_counts().index),
						 values=[x for x in df['SkinCancer'].value_counts()], hole=.35,
						 textinfo='label+percent', rotation=-45, marker_colors=colours),row=7, col=1)
	fig.update_layout(height=3200, font=dict(size=14), showlegend=False)
	fig.show()

def plot_frequency(df):
	fig, ax = plt.subplots(figsize = (14,6))
	sns.kdeplot(df[df["HeartDisease"]=='Yes']["Age"], alpha=1,shade = False, color="#ea4335", label="HeartDisease", ax = ax)
	sns.kdeplot(df[df["KidneyDisease"]=='Yes']["Age"], alpha=1,shade = False, color="#4285f4", label="KidneyDisease", ax = ax)
	sns.kdeplot(df[df["SkinCancer"]=='Yes']["Age"], alpha=1,shade = False, color="#fbbc05", label="SkinCancer", ax = ax)

	ax.set_xlabel("AgeCategory")
	ax.set_ylabel("Frequency")
	ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
	plt.savefig("figures/contin_frequency.png")
	plt.show()

def plot_bar(df):
	fig, ax = plt.subplots(figsize = (10,6))
	ax.hist(df[df["HeartDisease"]=='No']["GenHealth"], bins=10, alpha=0.8, color="#4285f4", label="No HeartDisease")
	ax.hist(df[df["HeartDisease"]=='Yes']["GenHealth"], bins=10, alpha=1, color="#ea4335", label="HeartDisease")
	ax.set_xlabel("GenHealth")
	ax.set_ylabel("Frequency")
	ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
	plt.savefig("figures/heart_disease.png")
	plt.show()
