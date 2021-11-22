import pandas as pd
import numpy as np
import plotly.express as px


class Base:

    def __init__(self, csv1, csv2, csv3, csv4):
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

    def add_condition(self, uponly=None, samebefore=None):
        if uponly is not None:
            self.csva = self.csva[self.csva['Up Day'] == uponly]
            self.csvb = self.csvb[self.csvb['Up Day'] == uponly]
            self.csvc = self.csvc[self.csvc['Up Day'] == uponly]
            self.csvd = self.csvd[self.csvd['Up Day'] == uponly]
            self.csvs = [self.csva, self.csvb, self.csvc, self.csvd]
            for i in range(len(self.names)):
                self.names[i] = self.names[i] + f'_{uponly}up'

        if samebefore is not None:
            self.csva = self.csva[self.csva['Same Direction'] == samebefore]
            self.csvb = self.csvb[self.csvb['Same Direction'] == samebefore]
            self.csvc = self.csvc[self.csvc['Same Direction'] == samebefore]
            self.csvd = self.csvd[self.csvd['Same Direction'] == samebefore]
            self.csvs = [self.csva, self.csvb, self.csvc, self.csvd]
            for i in range(len(self.names)):
                self.names[i] = self.names[i] + f'_{samebefore}samebefore'

    # create df summarizing mean, std dev, min, max, and abs min of each dataset
    def summarize_change(self):
        columns = ['Name', 'Median', 'Mean', 'Std. Deviation', 'Min', 'Max', 'Absolute Min']
        change_table = pd.DataFrame(columns=columns)
        for i in range(len(self.csvs)):
            median = round(self.csvs[i]['Pct Change'].median(), 6)
            mean = round(self.csvs[i]['Pct Change'].mean(), 6)
            std_dev = round(self.csvs[i]['Pct Change'].std(), 6)
            min = round(self.csvs[i]['Pct Change'].min(), 6)
            max = round(self.csvs[i]['Pct Change'].max(), 6)
            abs_min = round(self.csvs[i]['Pct Change'].abs().min(), 6)
            temp = {'Name': self.names[i], 'Median': median, 'Mean': mean, 'Std. Deviation': std_dev, 'Min': min,
                    'Max': max,
                    'Absolute Min': abs_min}
            change_table = change_table.append(temp, ignore_index=True)
        return change_table

    # create df summarizing mean, std dev, min, max, and quartiles of Volume
    # NOTE: ZERO VALUES ARE OMITTED FROM CALCULATIONS
    def summarize_volume(self):
        columns = ['Name', 'Median', 'Mean', 'Std. Deviation', 'Min', 'Max',
                   '25th percentile', '50th percentile', '75th percentile']
        volume_table = pd.DataFrame(columns=columns)
        for i in range(len(self.csvs)):
            nozero = self.csvs[i]['Volume'].replace(0, np.nan)
            median = nozero.median()
            mean = nozero.mean()
            std_dev = nozero.std()
            min = nozero.min()
            max = nozero.max()
            twentyfif = nozero.quantile(.25)
            fidyth = nozero.quantile(.5)
            sevendyfif = nozero.quantile(.75)
            temp = {'Name': self.names[i], 'Median': median, 'Mean': mean, 'Std. Deviation': std_dev, 'Min': min,
                    'Max': max,
                    '25th percentile': twentyfif, '50th percentile': fidyth, '75th percentile': sevendyfif}
            volume_table = volume_table.append(temp, ignore_index=True)
        return volume_table

    # summarizes mean + std dev of binomial variables
    def summarize_binomial(self):
        columns = ['Name', 'Pct. Up Day', 'Up Day Std. Deviation', 'Pct. Same Direct.', 'Same Direct. Std. Deviation']
        binomial_table = pd.DataFrame(columns=columns)
        for i in range(len(self.csvs)):
            pct_up = round(self.csvs[i]['Up Day'].mean(), 4)
            std_up = round(self.csvs[i]['Up Day'].std(), 4)
            pct_same = round(self.csvs[i]['Same Direction'].mean(), 4)
            std_same = round(self.csvs[i]['Same Direction'].std(), 4)
            temp = {'Name': self.names[i], 'Pct. Up Day': pct_up, 'Up Day Std. Deviation': std_up,
                    'Pct. Same Direct.': pct_same, 'Same Direct. Std. Deviation': std_same}
            binomial_table = binomial_table.append(temp, ignore_index=True)
        return binomial_table

    # histogram of change
    def change_histogram(self):
        for i in range(len(self.csvs)):
            fig = px.histogram(self.csvs[i]['Pct Change'], x='Pct Change', title=f'{self.names[i]} Pct. Change')
            fig.write_image(f'{self.names[i]}_histogram_change.png')

    # histogram of volume
    # NOTE: ZERO VALUES ARE OMITTED FROM COUNT
    def volume_histogram(self):
        for i in range(len(self.csvs)):
            df = self.csvs[i]['Volume'].replace(0, np.nan)
            fig = px.histogram(df, x='Volume', title=f'{self.names[i]} Volume')
            fig.write_image(f'{self.names[i]}_volume_change.png')

    def correlation(self):
        newone = self.combined.filter(
            ['Volume_^DJI', 'Pct Change_^DJI', 'Up Day_^DJI', 'Same Direction_^DJI', 'Volume_^GSPC',
             'Pct Change_^GSPC', 'Up Day_^GSPC', 'Same Direction_^GSPC', 'Volume_^NYA',
             'Pct Change_^NYA', 'Up Day_^NYA', 'Same Direction_^NYA', 'Volume_^IXIC',
             'Pct Change_^IXIC', 'Up Day_^IXIC', 'Same Direction_^IXIC'], axis=1)
        newone.corr().to_csv('correlation.csv')


# main method
def main():
    # initialize obj DJI = dow, GSPC = S&P500, NYA = NYSE, IXIC = NASDAQ
    dataset = Base(csv1='^DJI', csv2='^GSPC', csv3='^NYA', csv4='^IXIC')

    # downloads summaries as csv
    # dataset.summarize_change().to_csv('change.csv')
    # dataset.summarize_volume().to_csv('volume.csv')
    # dataset.summarize_binomial().to_csv('binomial.csv')

    # dataset.change_histogram()
    # dataset.volume_histogram()

    dataset.ExIndcorr_ot()

    # downloads combined dataset
    dataset.combined.to_csv('complete_set.csv')


########################################################################################################################


main()
