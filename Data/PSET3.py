import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

class Research:

    def __init__(self, csv1, csv2, csv3, csv4, window=25):
        self.names = [csv1, csv2, csv3, csv4]
        self.csva = pd.read_csv(f'{csv1}.csv')
        self.csvb = pd.read_csv(f'{csv2}.csv')
        self.csvc = pd.read_csv(f'{csv3}.csv')
        self.csvd = pd.read_csv(f'{csv4}.csv')
        self.csvs = [self.csva, self.csvb, self.csvc, self.csvd]

        # find avg price, pct change, if traded up (1,0), if traded in same direction as previous session (1,0)
        for csv in self.csvs:
            csv['Avg Price'] = (csv['High'] + csv['Low'] + csv['Close']) / 3
            csv['Pct Change'] = csv['Avg Price'].pct_change()
            csv['Up Day'] = (csv['Pct Change'] > 0) * 1
            csv['Same Direction'] = (csv['Up Day'] == csv['Up Day'].shift()) * 1

        # merge all datasets into one
        temp1 = pd.merge(self.csva, self.csvb, on='Date', suffixes=(f'_{csv1}', f'_{csv2}'))
        temp2 = pd.merge(self.csvc, self.csvd, on='Date', suffixes=(f'_{csv3}', f'_{csv4}'))
        self.combined = pd.merge(temp1, temp2, on='Date')

        # trim all individual datasets to same size
        length = len(self.combined)
        self.csva = self.csva.tail(length)
        self.csvb = self.csvb.tail(length)
        self.csvc = self.csvc.tail(length)
        self.csvd = self.csvd.tail(length)
        self.csvs = [self.csva, self.csvb, self.csvc, self.csvd]

        newdf = self.combined.filter(['Pct Change_^DJI', 'Pct Change_^GSPC',
                                      'Pct Change_^NYA', 'Pct Change_^IXIC'], axis=1)
        newerdf = newdf.rolling(window).corr()

        #copy to new df
        self.DJIdf = newerdf.filter(like='Pct Change_^DJI', axis=0).copy()
        self.GSPCdf = newerdf.filter(like='Pct Change_^GSPC', axis=0).copy()
        self.NYAdf = newerdf.filter(like='Pct Change_^NYA', axis=0).copy()
        self.IXICdf = newerdf.filter(like='Pct Change_^IXIC', axis=0).copy()

        #remove null/NaN values
        self.DJIdf['Pct Change_^DJI'] = self.DJIdf['Pct Change_^DJI'].replace('', np.nan)
        self.GSPCdf['Pct Change_^DJI'] = self.GSPCdf['Pct Change_^DJI'].replace('', np.nan)
        self.NYAdf['Pct Change_^DJI'] = self.NYAdf['Pct Change_^DJI'].replace('', np.nan)
        self.IXICdf['Pct Change_^DJI'] = self.IXICdf['Pct Change_^DJI'].replace('', np.nan)
        self.DJIdf.dropna(subset=['Pct Change_^GSPC'], inplace=True)
        self.GSPCdf.dropna(subset=['Pct Change_^DJI'], inplace=True)
        self.NYAdf.dropna(subset=['Pct Change_^IXIC'], inplace=True)
        self.IXICdf.dropna(subset=['Pct Change_^NYA'], inplace=True)

        self.DJIdf['Volume'] = self.combined['Volume_^DJI'].tail(len(self.DJIdf['Pct Change_^DJI'])).to_numpy()
        self.GSPCdf['Volume'] = self.combined['Volume_^GSPC'].tail(len(self.GSPCdf['Pct Change_^GSPC'])).to_numpy()
        self.NYAdf['Volume'] = self.combined['Volume_^NYA'].tail(len(self.NYAdf['Pct Change_^NYA'])).to_numpy()
        self.IXICdf['Volume'] = self.combined['Volume_^IXIC'].tail(len(self.IXICdf['Pct Change_^IXIC'])).to_numpy()


        #download rolling correlations to csv
        self.DJIdf.to_csv('DJIcorot.csv')
        self.GSPCdf.to_csv('GSPCcorot.csv')
        self.NYAdf.to_csv('NYAcorot.csv')
        self.IXICdf.to_csv('IXICcorot.csv')

    def generate(self):
        inex = ['GSPC', 'DJI', 'IXIC', 'NYA']
        dji = self.DJIdf
        gspc = self.GSPCdf
        ixic = self.IXICdf
        nya = self.NYAdf

        #create df of avg price(for lin regression generation later)
        pgdf = self.combined.filter(['Date', 'Avg Price_^GSPC', 'Avg Price_^DJI', 'Avg Price_^IXIC', 'Avg Price_^NYA'])
        pgdf.columns = ['Date', 'GSPC', 'DJI', 'IXIC', 'NYA']

        #index of trading day from start
        self.DJIdf['Index'] = np.arange(len(self.DJIdf))
        self.GSPCdf['Index'] = np.arange(len(self.GSPCdf))
        self.IXICdf['Index'] = np.arange(len(self.IXICdf))
        self.NYAdf['Index'] = np.arange(len(self.NYAdf))

        for un in inex:
            #find slope + intercept of linear regression
            fig = go.Figure(data=go.Scatter(y=dji[f'Pct Change_^{un}'], name=f'DJI-{un}'))
            cov = self.DJIdf.cov()['Index'][f'Pct Change_^{un}']
            var = self.DJIdf.var()['Index']
            slope = cov / var
            intercept = self.DJIdf[f'Pct Change_^{un}'].mean() - (self.DJIdf['Index'].mean() * slope)
            #create regression values to plot by multiplying it to index
            regression = self.DJIdf['Index'] * slope + intercept
            #create visuals
            fig.add_trace(
                go.Scatter(y=regression, name=f'Correlation Lin-Reg (y={round(slope, 6)}x + {round(intercept, 4)})',
                           line_color='#ffa500'))
            fig.update_layout(title=f"DJI-{un} Correlation Graph")
            fig.write_image(f'DJI-{un} Graph.png')
            fig.add_trace(go.Scatter(y=pgdf[un]/pgdf[un].max(), name=un))
            fig.write_image(f'DJI-{un} Graph {un} scaled overlay.png')
            fig = go.Figure(data=go.Histogram(x=dji[f'Pct Change_^{un}'], name=f'DJI-{un}'))
            fig.update_layout(title=f"DJI-{un} Histogram")
            fig.write_image(f'DJI-{un} Hist.png')

            # find slope + intercept of linear regression
            fig = go.Figure(data=go.Scatter(y=gspc[f'Pct Change_^{un}'], name=f'GSPC-{un}'))
            cov = self.GSPCdf.cov()['Index'][f'Pct Change_^{un}']
            var = self.GSPCdf.var()['Index']
            slope = cov / var
            intercept = self.GSPCdf[f'Pct Change_^{un}'].mean() - (self.GSPCdf['Index'].mean() * slope)
            # create regression values to plot by multiplying it to index
            regression = self.GSPCdf['Index'] * slope + intercept
            # create visuals
            fig.add_trace(
                go.Scatter(y=regression, name=f'Correlation Lin-Reg (y={round(slope, 6)}x + {round(intercept, 4)})',
                           line_color='#ffa500'))
            fig.update_layout(title=f"GSPC-{un} Correlation Graph")
            fig.write_image(f'GSPC-{un} Graph.png')
            fig.add_trace(go.Scatter(y=pgdf[un]/pgdf[un].max(), name=un))
            fig.write_image(f'GSPC-{un} Graph {un} scaled overlay.png')
            fig = go.Figure(data=go.Histogram(x=gspc[f'Pct Change_^{un}'], name=f'GSPC-{un}'))
            fig.update_layout(title=f"GSPC-{un} Histogram")
            fig.write_image(f'GSPC-{un} Hist.png')

            # find slope + intercept of linear regression
            fig = go.Figure(data=go.Scatter(y=ixic[f'Pct Change_^{un}'], name=f'IXIC-{un}'))
            cov = self.IXICdf.cov()['Index'][f'Pct Change_^{un}']
            var = self.IXICdf.var()['Index']
            slope = cov / var
            intercept = self.IXICdf[f'Pct Change_^{un}'].mean() - (self.IXICdf['Index'].mean() * slope)
            # create regression values to plot by multiplying it to index
            regression = self.IXICdf['Index'] * slope + intercept
            # create visuals
            fig.add_trace(
                go.Scatter(y=regression, name=f'Correlation Lin-Reg (y={round(slope, 6)}x + {round(intercept, 4)})',
                           line_color='#ffa500'))
            fig.update_layout(title=f"IXIC-{un} Correlation Graph")
            fig.write_image(f'IXIC-{un} Graph.png')
            fig.add_trace(go.Scatter(y=pgdf[un]/pgdf[un].max(), name=un))
            fig.write_image(f'IXIC-{un} Graph {un} scaled overlay.png')
            fig = go.Figure(data=go.Histogram(x=ixic[f'Pct Change_^{un}'], name=f'IXIC-{un}'))
            fig.update_layout(title=f"IXIC-{un} Histogram")
            fig.write_image(f'IXIC-{un} Hist.png')

            # find slope + intercept of linear regression
            fig = go.Figure(data=go.Scatter(y=nya[f'Pct Change_^{un}'], name=f'NYA-{un}'))
            cov = self.NYAdf.cov()['Index'][f'Pct Change_^{un}']
            var = self.NYAdf.var()['Index']
            slope = cov / var
            intercept = self.NYAdf[f'Pct Change_^{un}'].mean() - (self.NYAdf['Index'].mean() * slope)
            # create regression values to plot by multiplying it to index
            regression = self.NYAdf['Index'] * slope + intercept
            # create visuals
            fig.add_trace(
                go.Scatter(y=regression, name=f'Correlation Lin-Reg (y={round(slope, 6)}x + {round(intercept, 4)})',
                           line_color='#ffa500'))
            fig.update_layout(title=f"NYA-{un} Correlation Graph")
            fig.write_image(f'NYA-{un} Graph.png')
            fig.add_trace(go.Scatter(y=pgdf[un]/pgdf[un].max(), name=un))
            fig.write_image(f'NYA-{un} Graph {un} scaled overlay.png')
            fig = go.Figure(data=go.Histogram(x=nya[f'Pct Change_^{un}'], name=f'NYA-{un}'))
            fig.update_layout(title=f"NYA-{un} Histogram")
            fig.write_image(f'NYA-{un} Hist.png')

    def reganal(self):

        self.DJIdf['Index'] = np.arange(len(self.DJIdf))
        self.GSPCdf['Index'] = np.arange(len(self.GSPCdf))
        self.IXICdf['Index'] = np.arange(len(self.IXICdf))
        self.NYAdf['Index'] = np.arange(len(self.NYAdf))

        independent = 'Index'
        dependent = 'Pct Change_^GSPC'
        X = self.DJIdf[independent]
        X = sm.add_constant(X)
        Y = self.DJIdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())

        dependent = 'Pct Change_^NYA'
        X = self.DJIdf.loc[:, independent]
        X = sm.add_constant(X)
        Y = self.DJIdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())

        dependent = 'Pct Change_^IXIC'
        X = self.DJIdf.loc[:, independent]
        X = sm.add_constant(X)
        Y = self.DJIdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())

        dependent = 'Pct Change_^NYA'
        X = self.GSPCdf.loc[:, independent]
        X = sm.add_constant(X)
        Y = self.GSPCdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())

        dependent = 'Pct Change_^IXIC'
        X = self.GSPCdf.loc[:, independent]
        X = sm.add_constant(X)
        Y = self.GSPCdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())

        dependent = 'Pct Change_^NYA'
        X = self.IXICdf.loc[:, independent]
        X = sm.add_constant(X)
        Y = self.IXICdf.loc[:, dependent]
        result = sm.OLS(Y, X).fit()
        print(result.summary())


    #Generate Standard Deviations of all correlations over time
    def genmore(self):
        dji = self.DJIdf
        gspc = self.GSPCdf
        ixic = self.IXICdf
        nya = self.NYAdf
        a = dji.std()
        a.to_csv('djistd.csv')
        b = gspc.std()
        b.to_csv('gspcstd.csv')
        c = ixic.std()
        c.to_csv('ixicstd.csv')
        d = nya.std()
        d.to_csv('nyastd.csv')






dataset = Research(csv1='^DJI', csv2='^GSPC', csv3='^NYA', csv4='^IXIC')
dataset.reganal()
#dataset.generate()
#dataset.genmore()
