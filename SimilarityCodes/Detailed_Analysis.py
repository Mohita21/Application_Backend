

def DeResults_Similarity_Output(y1,y2,dates):
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    Yield = pd.DataFrame({'y1': y1}, index=pd.DatetimeIndex(dates))
    Yield['y2'] = y2
    List_Prc = ['Observation', 'Trend', 'Seasonality', 'Residual(Noise)']
    Prc = np.zeros((len(List_Prc)))
    Figure_dpi = 600
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    # ****** Time series plot Function *******************************************************************************
    def tsplot(Yield, Y_axis_Label, X_axis_Label, pp):
        import pandas as pd
        from scipy import stats
        from matplotlib.backends.backend_pdf import PdfPages
        Yield.index = pd.to_datetime(Yield.index)
        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%Y')

        fig = plt.figure(dpi=Figure_dpi)
        axis1 = fig.add_subplot(211)
        axis1.plot(Yield.iloc[:, 0], ls='-', lw='1', c='black')
        axis1.plot(Yield.iloc[:, 1], ls='--', lw='.6', c='orangered')
        axis1.xaxis.set_major_formatter(myFmt)
        axis1.legend(Yield.columns[0:Yield.shape[1] + 1], loc='best', fontsize=7)
        axis1.set_xlabel(X_axis_Label, fontsize=8)
        axis1.set_ylabel(Y_axis_Label, fontsize=8)
        axis1.set_title('Time series Plot', fontsize=15)

        Yield_normalized = Yield
        Y_min = np.min(Yield)
        Y_max = np.max(Yield)

        for j in range(Yield_normalized.shape[1]):
            for k in range(Yield_normalized.shape[0]):
                Yield_normalized = (Yield - Y_min) / (Y_max - Y_min)

        # print(Yield_normalized)

        axis2 = fig.add_subplot(212)
        axis2.plot(Yield_normalized.iloc[:, 0], ls='-', lw='1', c='black')
        axis2.plot(Yield_normalized.iloc[:, 1], ls='--', lw='.6', c='orangered')
        axis2.xaxis.set_major_formatter(myFmt)
        axis2.legend(Yield.columns[0:Yield.shape[1] + 1], loc='best', fontsize=7)
        axis2.set_xlabel(X_axis_Label, fontsize=8)
        axis2.set_ylabel('Normalized {}'.format(Y_axis_Label), fontsize=8)

        fig.tight_layout()
        # fig.savefig('TimeSeries_Plot.jpeg')
        # pp = PdfPages("Similarity_Output_Plots.pdf")
        pp.savefig(fig, dpi=300, transparent=True)
        plt.close()

    # ****** Applying decomposition technique to illustrate unobserved components of time series ***************************
    def decomposition(Yield, pp):
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_pdf import PdfPages
        myFmt = mdates.DateFormatter('%Y-%m')

        Decompse_result = np.zeros((int(Yield.shape[0] / 30), 5, Yield.shape[1]))
        Yield.index = pd.to_datetime(Yield.index)
        from statsmodels.tsa.seasonal import seasonal_decompose

        for i in range(Yield.shape[1]):
            series = Yield.iloc[:, i]
            resample = series.resample('M')
            monthly_mean = resample.mean()
            result = seasonal_decompose(monthly_mean, model='multiplicative')  # multiplicative  or additive
            Decompse_result[:, 0, i] = result.observed.values
            Decompse_result[:, 1, i] = result.trend.values
            Decompse_result[:, 2, i] = result.seasonal.values
            Decompse_result[:, 3, i] = result.resid.values
            # print(Yield.columns[i])
            # result.plot()
            # plt.show()

        Xrng = result.observed.index

        fig1 = plt.figure(figsize=(8, 8), dpi=Figure_dpi)  # figsize=(8,2)

        axis1 = fig1.add_subplot(411)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis1.plot(Xrng, Decompse_result[:, 0, i], ls='-', lw='1', c='black')
            else:
                axis1.plot(Xrng, Decompse_result[:, 0, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis1.set_xlim(min(Xrng), max(Xrng))
        axis1.xaxis.set_major_formatter(myFmt)
        axis1.legend(Yield.columns[0:2], loc='best', fontsize=7)  # ls='--',lw='.6',c='orangered')
        # axis1.set_xlabel('Time (month)',fontsize = 8)
        axis1.set_ylabel('Observation', fontsize=8)
        axis1.set_title('Time series Decomposition', fontsize=15)

        axis2 = fig1.add_subplot(412)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis2.plot(Xrng, Decompse_result[:, 1, i], ls='-', lw='1', c='black')
            else:
                axis2.plot(Xrng, Decompse_result[:, 1, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis2.set_xlim(min(Xrng), max(Xrng))
        axis2.xaxis.set_major_formatter(myFmt)
        axis2.legend(Yield.columns[0:2], loc='best', fontsize=7)
        # axis2.set_xlabel('Time (month)',fontsize = 8)
        axis2.set_ylabel('Trend', fontsize=8)

        axis3 = fig1.add_subplot(413)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis3.plot(Xrng, Decompse_result[:, 2, i], ls='-', lw='1', c='black')
            else:
                axis3.plot(Xrng, Decompse_result[:, 2, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis3.set_xlim(min(Xrng), max(Xrng))
        axis3.xaxis.set_major_formatter(myFmt)
        axis3.legend(Yield.columns[0:2], loc='best', fontsize=7)
        # axis3.set_xlabel('Time (month)',fontsize = 8)
        axis3.set_ylabel('Seasonality', fontsize=8)

        axis4 = fig1.add_subplot(414)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis4.plot(Xrng, Decompse_result[:, 3, i], ls='-', lw='1', c='black')
            else:
                axis4.plot(Xrng, Decompse_result[:, 3, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis4.set_xlim(min(Xrng), max(Xrng))
        axis4.xaxis.set_major_formatter(myFmt)
        axis4.legend(Yield.columns[0:2], loc='best', fontsize=7)
        axis4.set_xlabel('Time (month)', fontsize=8)
        axis4.set_ylabel('Residual', fontsize=8)

        # fig1.savefig('TimeSeries_Decomposition.jpeg')

        pp.savefig(fig1, dpi=300, transparent=True)
        plt.close()

    # ************* Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique **************
    def DTW_of_components(Yield, f):
        Decompse_result = np.zeros((int(Yield.shape[0] / 30), 5, Yield.shape[1]))
        Yield.index = pd.to_datetime(Yield.index)
        from statsmodels.tsa.seasonal import seasonal_decompose

        for i in range(2):
            series = Yield.iloc[:, i]
            resample = series.resample('M')
            monthly_mean = resample.mean()
            result = seasonal_decompose(monthly_mean, model='multiplicative')  # multiplicative  or additive
            Decompse_result[:, 0, i] = result.observed.values
            Decompse_result[:, 1, i] = result.trend.values
            Decompse_result[:, 2, i] = result.seasonal.values
            Decompse_result[:, 3, i] = result.resid.values

        # !pip install fastdtw
        from scipy import stats
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        list = ['Observation', 'Trend', 'Seasonality', 'Residual']
        where_are_NaNs = np.isnan(Decompse_result)
        Decompse_result[where_are_NaNs] = 0
        A = np.zeros((Decompse_result.shape[0], Decompse_result.shape[2]))
        for n in range(4):
            for m in range(Decompse_result.shape[2]):
                A[:, m] = Decompse_result[:, n, m]

            A_az = stats.zscore(A,
                                axis=0)  # Calculates the z score of each value in the sample, relative to the sample mean and standard deviation

            Dis_az = np.zeros((A.shape[1], A.shape[1]))
            Dis_az = pd.DataFrame(Dis_az, index=Yield.columns, columns=Yield.columns)

            for i in range(A.shape[1]):
                x = A_az[:, i]  # /np.max(Yield.iloc[:,i].values)
                for j in range(A.shape[1]):
                    y = A_az[:, j]  # /np.max(Yield.iloc[:,j].values)
                    distance, path = fastdtw(x, y, dist=euclidean)
                    Dis_az.iloc[i, j] = round(distance, 2)

            # ** Thresholds ***
            if Dis_az.iloc[0, 1] <= 1:
                Prc[n] = 100
            elif 1 < Dis_az.iloc[0, 1] <= 10:
                Prc[n] = 90
            elif 10 < Dis_az.iloc[0, 1] <= 20:
                Prc[n] = 85
            elif 20 < Dis_az.iloc[0, 1] <= 30:
                Prc[n] = 80
            elif 30 < Dis_az.iloc[0, 1] <= 40:
                Prc[n] = 75
            elif 40 < Dis_az.iloc[0, 1] <= 50:
                Prc[n] = 70
            elif 50 < Dis_az.iloc[0, 1] <= 60:
                Prc[n] = 65
            elif 60 < Dis_az.iloc[0, 1] <= 70:
                Prc[n] = 60
            else:
                Prc[n] = 55

            # Prc[0,n]=Dis_az.iloc[0,1]
            print('\n ***** Distance Matrix for', list[n], '*****', file=f)
            print(Dis_az, file=f)
            # print("\n \n")

        from tabulate import tabulate
        my_df = pd.DataFrame(np.transpose(Prc), index=List_Prc, columns=['Percentage of Similarity'])
        # print("\n \n",my_df.head(4))
        print("\n \n", tabulate(my_df.head(4), headers='keys', tablefmt='psql'), file=f)

    # ********* Detail Analysis Output ************************
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages("/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Output_Plots.pdf")
    f = open('/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Output_Results.txt', 'w')

    # print('Time series Plot')
    Y_axis_Label = 'Yield (pound/acre)'  # input ('Enter Y-axis-label for time series plot (e.g. Yield (pound/acre)):')  # Examples: Yield (pound/acre) or Price (CAD per pound)
    X_axis_Label = 'Time(day)'  # input('Enter X-axis-label for time series plot (e.g. Time(day)):') # Examples: day or week or month
    tsplot(Yield, Y_axis_Label, X_axis_Label, pp)

    # print('Applying decomposition technique to illustrate unobserved components of time series')
    decomposition(Yield, pp)  # for daily time series

    pp.close()

    from termcolor import colored
    # print(colored('\n \n Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique \n', 'red', attrs=['bold']),file=f)
    print('\n \n # Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique: \n',
          file=f)
    # print('\n \n Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique')
    DTW_of_components(Yield, f)  # for daily time series
    f.close()


# In[ ]:


def DeResults_Similarity_Input(StartDate, county_1_conditions, county_2_conditions):
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    Figure_dpi = 600
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    date = np.array(StartDate, dtype=np.datetime64)  # '2015-07-04'
    date = date + np.arange(county_1_conditions.shape[0])
    county_1_conditions.index = date
    county_2_conditions.index = date

    List_Prc = ['Observation', 'Trend', 'Seasonality', 'Residual(Noise)', 'Linear Correlation']
    Prc = np.zeros((len(List_Prc)))

    # ****** Time series plot Function *******************************************************************************
    def tsplot(Yield, Y_axis_Label, X_axis_Label, pp):
        import pandas as pd
        from scipy import stats
        from matplotlib.backends.backend_pdf import PdfPages
        Yield.index = pd.to_datetime(Yield.index)
        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%Y')

        fig = plt.figure(dpi=Figure_dpi)
        axis1 = fig.add_subplot(211)
        axis1.plot(Yield.iloc[:, 0], ls='-', lw='1', c='black')
        axis1.plot(Yield.iloc[:, 1], ls='--', lw='.6', c='orangered')
        axis1.xaxis.set_major_formatter(myFmt)
        axis1.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)
        axis1.set_xlabel(X_axis_Label, fontsize=8)
        axis1.set_ylabel(Y_axis_Label, fontsize=8)
        axis1.set_title('Time series Plot for {}'.format(Yield.columns[0]), fontsize=15)

        Yield_normalized = Yield
        Y_min = np.min(Yield)
        Y_max = np.max(Yield)

        for j in range(Yield_normalized.shape[1]):
            for k in range(Yield_normalized.shape[0]):
                Yield_normalized = (Yield - Y_min) / (Y_max - Y_min)

        # print(Yield_normalized)

        axis2 = fig.add_subplot(212)
        axis2.plot(Yield_normalized.iloc[:, 0], ls='-', lw='1', c='black')
        axis2.plot(Yield_normalized.iloc[:, 1], ls='--', lw='.6', c='orangered')
        axis2.xaxis.set_major_formatter(myFmt)
        axis2.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)
        axis2.set_xlabel(X_axis_Label, fontsize=8)
        axis2.set_ylabel('Normalized {}'.format(Y_axis_Label), fontsize=8)

        fig.tight_layout()
        # fig.savefig('TimeSeries_Plot.jpeg')
        # pp = PdfPages("Similarity_Output_Plots.pdf")
        pp.savefig(fig, dpi=300, transparent=True)
        plt.close()

    # ****** Applying decomposition technique to illustrate unobserved components of time series ***************************
    def decomposition(Yield, pp):
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_pdf import PdfPages
        myFmt = mdates.DateFormatter('%Y-%m')

        Decompse_result = np.zeros((int(Yield.shape[0] / 30), 5, Yield.shape[1]))
        Yield.index = pd.to_datetime(Yield.index)
        from statsmodels.tsa.seasonal import seasonal_decompose

        for i in range(Yield.shape[1]):
            series = Yield.iloc[:, i]
            resample = series.resample('M')
            monthly_mean = resample.mean()
            result = seasonal_decompose(monthly_mean, model='multiplicative')  # multiplicative  or additive
            Decompse_result[:, 0, i] = result.observed.values
            Decompse_result[:, 1, i] = result.trend.values
            Decompse_result[:, 2, i] = result.seasonal.values
            Decompse_result[:, 3, i] = result.resid.values
            # print(Yield.columns[i])
            # result.plot()
            # plt.show()

        Xrng = result.observed.index

        fig1 = plt.figure(figsize=(8, 8), dpi=Figure_dpi)  # figsize=(8,2)

        axis1 = fig1.add_subplot(411)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis1.plot(Xrng, Decompse_result[:, 0, i], ls='-', lw='1', c='black')
            else:
                axis1.plot(Xrng, Decompse_result[:, 0, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis1.set_xlim(min(Xrng), max(Xrng))
        axis1.xaxis.set_major_formatter(myFmt)
        axis1.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)  # ls='--',lw='.6',c='orangered')
        # axis1.set_xlabel('Time (month)',fontsize = 8)
        axis1.set_ylabel('Observation', fontsize=8)
        axis1.set_title('Time series Decomposition for {}'.format(Yield.columns[0]), fontsize=15)

        axis2 = fig1.add_subplot(412)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis2.plot(Xrng, Decompse_result[:, 1, i], ls='-', lw='1', c='black')
            else:
                axis2.plot(Xrng, Decompse_result[:, 1, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis2.set_xlim(min(Xrng), max(Xrng))
        axis2.xaxis.set_major_formatter(myFmt)
        axis2.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)
        # axis2.set_xlabel('Time (month)',fontsize = 8)
        axis2.set_ylabel('Trend', fontsize=8)

        axis3 = fig1.add_subplot(413)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis3.plot(Xrng, Decompse_result[:, 2, i], ls='-', lw='1', c='black')
            else:
                axis3.plot(Xrng, Decompse_result[:, 2, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis3.set_xlim(min(Xrng), max(Xrng))
        axis3.xaxis.set_major_formatter(myFmt)
        axis3.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)
        # axis3.set_xlabel('Time (month)',fontsize = 8)
        axis3.set_ylabel('Seasonality', fontsize=8)

        axis4 = fig1.add_subplot(414)
        for i in range(Decompse_result.shape[2]):  # (round(0.2+Yield.shape[1]/2)):
            if i == 0:
                axis4.plot(Xrng, Decompse_result[:, 3, i], ls='-', lw='1', c='black')
            else:
                axis4.plot(Xrng, Decompse_result[:, 3, i], ls='--', lw='.6',
                           c='orangered')  # or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
        axis4.set_xlim(min(Xrng), max(Xrng))
        axis4.xaxis.set_major_formatter(myFmt)
        axis4.legend(['County_1_{}'.format(Yield.columns[0]), 'County_2_{}'.format(Yield.columns[0])], loc='best',
                     fontsize=7)
        axis4.set_xlabel('Time (month)', fontsize=8)
        axis4.set_ylabel('Residual', fontsize=8)

        # fig1.savefig('TimeSeries_Decomposition.jpeg')

        pp.savefig(fig1, dpi=300, transparent=True)
        plt.close()

    # ************* Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique **************
    def DTW_of_components(Yield, f):
        Decompse_result = np.zeros((int(Yield.shape[0] / 30), 5, Yield.shape[1]))
        Yield.index = pd.to_datetime(Yield.index)
        from statsmodels.tsa.seasonal import seasonal_decompose

        for i in range(2):
            series = Yield.iloc[:, i]
            resample = series.resample('M')
            monthly_mean = resample.mean()
            result = seasonal_decompose(monthly_mean, model='multiplicative')  # multiplicative  or additive
            Decompse_result[:, 0, i] = result.observed.values
            Decompse_result[:, 1, i] = result.trend.values
            Decompse_result[:, 2, i] = result.seasonal.values
            Decompse_result[:, 3, i] = result.resid.values

        # !pip install fastdtw
        from scipy import stats
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        list = ['Observation', 'Trend', 'Seasonality', 'Residual']
        where_are_NaNs = np.isnan(Decompse_result)
        Decompse_result[where_are_NaNs] = 0
        A = np.zeros((Decompse_result.shape[0], Decompse_result.shape[2]))
        for n in range(4):
            for m in range(Decompse_result.shape[2]):
                A[:, m] = Decompse_result[:, n, m]

            A_az = stats.zscore(A,
                                axis=0)  # Calculates the z score of each value in the sample, relative to the sample mean and standard deviation

            Dis_az = np.zeros((A.shape[1], A.shape[1]))
            Dis_az = pd.DataFrame(Dis_az, index=['County_1_{}'.format(Yield.columns[0]),
                                                 'County_2_{}'.format(Yield.columns[0])],
                                  columns=['County_1_{}'.format(Yield.columns[0]),
                                           'County_2_{}'.format(Yield.columns[0])])

            for i in range(A.shape[1]):
                x = A_az[:, i]  # /np.max(Yield.iloc[:,i].values)
                for j in range(A.shape[1]):
                    y = A_az[:, j]  # /np.max(Yield.iloc[:,j].values)
                    distance, path = fastdtw(x, y, dist=euclidean)
                    Dis_az.iloc[i, j] = round(distance, 2)

            # ** Thresholds ***
            if Dis_az.iloc[0, 1] <= 1:
                Prc[n] = 100
            elif 1 < Dis_az.iloc[0, 1] <= 10:
                Prc[n] = 90
            elif 10 < Dis_az.iloc[0, 1] <= 20:
                Prc[n] = 85
            elif 20 < Dis_az.iloc[0, 1] <= 30:
                Prc[n] = 80
            elif 30 < Dis_az.iloc[0, 1] <= 40:
                Prc[n] = 75
            elif 40 < Dis_az.iloc[0, 1] <= 50:
                Prc[n] = 70
            elif 50 < Dis_az.iloc[0, 1] <= 60:
                Prc[n] = 65
            elif 60 < Dis_az.iloc[0, 1] <= 70:
                Prc[n] = 60
            else:
                Prc[n] = 55

            # Prc[0,n]=Dis_az.iloc[0,1]
            print('\n ***** Distance Matrix for', list[n], '*****', file=f)
            print(Dis_az, file=f)
            # print("\n \n",file=f)

        from scipy.stats import pearsonr
        corr, _ = pearsonr(Yield.iloc[:, 0], Yield.iloc[:, 1])
        # print(corr)
        Prc[n + 1] = round(corr * 100)

        from tabulate import tabulate
        my_df = pd.DataFrame(np.transpose(Prc), index=List_Prc,
                             columns=['Percentage of Similarity for {}'.format(Yield.columns[0])])
        # print("\n \n",my_df.head(4))
        print("\n \n", tabulate(my_df.head(5), headers='keys', tablefmt='psql'), file=f)

    # ********* Detail Analysis Output ************************
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages("/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Inputss_Plots.pdf")
    f = open('/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Inputs_Results.txt', 'w')
    for CN in range(county_1_conditions.shape[1]):
        my_array = np.transpose(np.array([county_1_conditions.iloc[:, CN], county_2_conditions.iloc[:, CN]]))
        Yield = pd.DataFrame(my_array, index=county_1_conditions.index,
                             columns=[county_1_conditions.columns[CN], county_2_conditions.columns[CN]])

        # print('Time series Plot')
        Y_axis_Label = county_1_conditions.columns[
            CN]  # input ('Enter Y-axis-label for time series plot (e.g. Yield (pound/acre)):')  # Examples: Yield (pound/acre) or Price (CAD per pound)
        X_axis_Label = 'Time (day)'  # input('Enter X-axis-label for time series plot (e.g. Time(day)):') # Examples: day or week or month
        tsplot(Yield, Y_axis_Label, X_axis_Label, pp)

        # print('Applying decomposition technique to illustrate unobserved components of time series')
        decomposition(Yield, pp)  # for daily time series

        from termcolor import colored
        print(
            '\n \n # Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique ({}): \n'.format(
                Yield.columns[0]), file=f)
        # print('\n \n Percentage of Similarities for Unobserved components using Dynamic Time Warping Technique')
        DTW_of_components(Yield, f)  # for daily time series
    pp.close()
    f.close()


# In[ ]:


def Similarity_Input_Output(county_1_conditions, county_2_conditions, y1,y2,dates):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    Yield = pd.DataFrame({'y1': y1}, index=pd.DatetimeIndex(dates))
    Yield['y2'] = y2
    county_1_conditions.index = Yield.index
    county_2_conditions.index = Yield.index

    List_Prc = ['Average_Linear_correlation', 'Effective_Lag_time', 'Effective_Predictors']
    Prc = np.zeros((len(List_Prc)))

    # ****** Interstage_dependency_analysis to find Effective Lag-time of Predictors *******************************************************************************
    def interstage_dependency(county_1_conditions, county_2_conditions, Yield, pp):
        from numpy.random import randn
        from numpy.random import seed
        from scipy.stats import pearsonr
        import matplotlib
        import matplotlib.pyplot as plt
        Figure_dpi = 600
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        # seed random number generator
        seed(42)

        num = 70

        # calculate Pearson's correlation
        R = np.zeros((num, county_1_conditions.shape[1], Yield.shape[1]))
        for i in range(num):
            wlag = i * 5
            y = Yield.iloc[0 + wlag:Yield.shape[0]][:]
            for n in range(y.shape[1]):
                if n == 0:
                    Predictors = county_1_conditions
                else:
                    Predictors = county_2_conditions

                p = Predictors.iloc[0:Predictors.shape[0] - wlag][:]

                for j in range(p.shape[1]):
                    corr, _ = pearsonr(p.iloc[:, j], y.iloc[:, n])
                    # print('Pearsons correlation: %.3f' % corr)
                    R[i, j, n] = corr

        # plot
        fig = plt.figure(dpi=Figure_dpi)
        fig.set_figheight(((np.int(p.shape[1] / 3)) + 1 * 11))
        fig.set_figwidth(15)
        d = {}
        for i in range(p.shape[1]):
            d["axis" + str(i + 1)] = fig.add_subplot(np.int(p.shape[1] / 3) + 1, 3, i + 1)
            for j in range(y.shape[1]):
                if j == 0:
                    # plt.plot(np.arange(0,num,1)*5,R[:,i,j],ls='-',lw='1',c='black')
                    d["axis" + str(i + 1)].plot(np.arange(0, num, 1) * 5, R[:, i, j], ls='-', lw='1', c='black')
                else:
                    # plt.plot(np.arange(0,num,1)*5,R[:,i,j],ls='--',lw='.6',c='orangered') #or# ls=':',lw='1.5',marker='+',mew=0.3,c='limegreen')
                    d["axis" + str(i + 1)].plot(np.arange(0, num, 1) * 5, R[:, i, j], ls='--', lw='.6',
                                                c='orangered')  # ls='--',lw='.6',c='orangered')

            d["axis" + str(i + 1)].plot(np.arange(0, num, 1) * 5, np.zeros(num), ls='--')
            d["axis" + str(i + 1)].legend(y.columns, loc='best', fontsize=7)
            d["axis" + str(i + 1)].set_xlabel('Lag time (day)', fontsize=8)
            d["axis" + str(i + 1)].set_ylabel('Correlation', fontsize=8)
            d["axis" + str(i + 1)].set_title(p.columns[i])

        pp.savefig(fig, dpi=300, transparent=True)
        plt.close()

    def Effective_lag(county_1_conditions, county_2_conditions, Yield, f):

        from numpy.random import randn
        from numpy.random import seed
        from scipy.stats import pearsonr
        # seed random number generator
        seed(42)

        num = 70

        # calculate Pearson's correlation
        R = np.zeros((num, county_1_conditions.shape[1], Yield.shape[1]))
        for i in range(num):
            wlag = i * 5
            y = Yield.iloc[0 + wlag:Yield.shape[0]][:]
            for n in range(y.shape[1]):
                if n == 0:
                    Predictors = county_1_conditions
                else:
                    Predictors = county_2_conditions

                p = Predictors.iloc[0:Predictors.shape[0] - wlag][:]

                for j in range(p.shape[1]):
                    corr, _ = pearsonr(p.iloc[:, j], y.iloc[:, n])
                    # print('Pearsons correlation: %.3f' % corr)
                    R[i, j, n] = corr

        from termcolor import colored
        pd.set_option('display.max_columns', None)
        eff_lag = np.zeros((3, R.shape[1], R.shape[2]))
        pd.options.display.float_format = "{:,.2f}".format
        for j in range(R.shape[2]):
            for i in range(R.shape[1]):
                if max(abs(R[10:, i, j])) in R[:, i, j]:
                    eff_lag[0, i, j] = (np.argmax(abs(R[10:, i, j])) + 10) * 5
                    eff_lag[1, i, j] = round(eff_lag[0, i, j] / 30)
                    eff_lag[2, i, j] = max(abs(R[10:, i, j]))
                else:
                    eff_lag[0, i, j] = (np.argmax(abs(R[10:, i, j])) + 10) * 5
                    eff_lag[1, i, j] = round(eff_lag[0, i, j] / 30)
                    eff_lag[2, i, j] = -max(abs(R[10:, i, j]))
            # print(eff_lag[:,:,j])

        Cor_val = np.zeros((1, y.shape[1]))
        for i in range(y.shape[1]):
            # print(colored(y.columns[i], 'red', attrs=['bold'])),file=f
            print(y.columns[i], file=f)
            # print(y.columns[i])
            Elag = pd.DataFrame(eff_lag[:, :, i], index=['Best_daily_lag', 'Best_monthly_lag', 'Maximum_correlation'],
                                columns=[p.columns])
            print(Elag.head(), file=f)
            print("\n \n", file=f)

            Cor_val[0, i] = np.mean(np.sort(abs(Elag.iloc[2, :]))[::-1])

        A = Cor_val[0, 0] - Cor_val[0, 1]

        if A <= 0.01:
            PS = 90
        elif 0.01 < A <= 0.1:
            PS = 70
        elif 0.1 < A <= 0.2:
            PS = 50
        elif 0.2 < A <= 0.3:
            PS = 30
        else:
            PS = 10

        return (PS)

    def feature_selection(county_1_conditions, county_2_conditions, Yield, f):

        # Inter-stage dependency and Linear correlation
        from numpy.random import randn
        from numpy.random import seed
        from scipy.stats import pearsonr
        # seed random number generator
        seed(42)

        num = 70

        # calculate Pearson's correlation
        R = np.zeros((num, county_1_conditions.shape[1], Yield.shape[1]))
        for i in range(num):

            wlag = i * 5
            y = Yield.iloc[0 + wlag:Yield.shape[0]][:]
            for n in range(Yield.shape[1]):

                if n == 0:
                    Predictors = county_1_conditions
                else:
                    Predictors = county_2_conditions

                p = Predictors.iloc[0:Predictors.shape[0] - wlag][:]

                for j in range(p.shape[1]):
                    corr, _ = pearsonr(p.iloc[:, j], y.iloc[:, n])
                    # print('Pearsons correlation: %.3f' % corr)
                    R[i, j, n] = corr

        from termcolor import colored
        pd.set_option('display.max_columns', None)
        eff_lag = np.zeros((3, R.shape[1], R.shape[2]))
        pd.options.display.float_format = "{:,.2f}".format
        for j in range(R.shape[2]):
            for i in range(R.shape[1]):
                if max(abs(R[10:, i, j])) in R[:, i, j]:
                    eff_lag[0, i, j] = (np.argmax(abs(R[10:, i, j])) + 10) * 5
                    eff_lag[1, i, j] = round(eff_lag[0, i, j] / 30)
                    eff_lag[2, i, j] = max(abs(R[10:, i, j]))
                else:
                    eff_lag[0, i, j] = (np.argmax(abs(R[10:, i, j])) + 10) * 5
                    eff_lag[1, i, j] = round(eff_lag[0, i, j] / 30)
                    eff_lag[2, i, j] = -max(abs(R[10:, i, j]))
            # print(eff_lag[:,:,j])

        # Effective predictors and corresponding lag-time
        # Binary GA
        class BGA():
            """
            Simple 0-1 genetic algorithm.
            User Guide:
            >> test = GA(pop_shape=(10, 10), method=np.sum)
            >> solution, fitness = test.run()
            """

            def __init__(self, pop_shape, method, p_c=0.8, p_m=0.2, max_round=1000, early_stop_rounds=None,
                         verbose=None, maximum=True):
                """
                Args:
                    pop_shape: The shape of the population matrix.
                    method: User-defined medthod to evaluate the single individual among the population.
                            Example:
                            def method(arr): # arr is a individual array
                                return np.sum(arr)
                    p_c: The probability of crossover.
                    p_m: The probability of mutation.
                    max_round: The maximun number of evolutionary rounds.
                    early_stop_rounds: Default is None and must smaller than max_round.
                    verbose: 'None' for not printing progress messages. int type number for printing messages every n iterations.
                    maximum: 'True' for finding the maximum value while 'False' for finding the minimum value.
                """
                if early_stop_rounds != None:
                    assert (max_round > early_stop_rounds)
                self.pop_shape = pop_shape
                self.method = method
                self.pop = np.ones(pop_shape)
                self.fitness = np.zeros(pop_shape[0])
                self.p_c = p_c
                self.p_m = p_m
                self.max_round = max_round
                self.early_stop_rounds = early_stop_rounds
                self.verbose = verbose
                self.maximum = maximum

            def evaluation(self, pop):
                """
                Computing the fitness of the input popluation matrix.
                Args:
                    p: The population matrix need to be evaluated.
                """
                return np.array([self.method(i) for i in pop])

            def initialization(self):
                """
                Initalizing the population which shape is self.pop_shape(0-1 matrix).
                """
                self.pop = np.random.randint(low=0, high=2, size=self.pop_shape)
                self.fitness = self.evaluation(self.pop)

            def crossover(self, ind_0, ind_1):
                """
                Single point crossover.
                Args:
                    ind_0: individual_0
                    ind_1: individual_1
                Ret:
                    new_0, new_1: the individuals generatd after crossover.
                """
                assert (len(ind_0) == len(ind_1))

                point = np.random.randint(len(ind_0))
                #         new_0, new_1 = np.zeros(len(ind_0)),  np.zeros(len(ind_0))
                new_0 = np.hstack((ind_0[:point], ind_1[point:]))
                new_1 = np.hstack((ind_1[:point], ind_0[point:]))

                assert (len(new_0) == len(ind_0))
                return new_0, new_1

            def mutation(self, indi):
                """
                Simple mutation.
                Arg:
                    indi: individual to mutation.
                """
                point = np.random.randint(len(indi))
                indi[point] = 1 - indi[point]
                return indi

            def rws(self, size, fitness):
                """
                Roulette Wheel Selection.
                Args:
                    size: the size of individuals you want to select according to their fitness.
                    fitness: the fitness of population you want to apply rws to.
                """
                if self.maximum:
                    fitness_ = fitness
                else:
                    fitness_ = 1.0 / fitness
                #         fitness_ = fitness
                idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,
                                       p=fitness_ / fitness_.sum())  #
                return idx

            def run(self):
                """
                Run the genetic algorithm.
                Ret:
                    global_best_ind: The best indiviudal during the evolutionary process.
                    global_best_fitness: The fitness of the global_best_ind.
                """
                global_best = 0
                self.initialization()
                best_index = np.argsort(self.fitness)[0]
                global_best_fitness = self.fitness[best_index]
                global_best_ind = self.pop[best_index, :]
                eva_times = self.pop_shape[0]
                count = 0

                for it in range(self.max_round):
                    next_gene = []

                    for n in range(int(self.pop_shape[0] / 2)):
                        i, j = self.rws(2, self.fitness)  # choosing 2 individuals with rws.
                        indi_0, indi_1 = self.pop[i, :].copy(), self.pop[j, :].copy()
                        if np.random.rand() < self.p_c:
                            indi_0, indi_1 = self.crossover(indi_0, indi_1)

                        if np.random.rand() < self.p_m:
                            indi_0 = self.mutation(indi_0)
                            indi_1 = self.mutation(indi_1)

                        next_gene.append(indi_0)
                        next_gene.append(indi_1)

                    self.pop = np.array(next_gene)
                    self.fitness = self.evaluation(self.pop)
                    eva_times += self.pop_shape[0]

                    if self.maximum:
                        if np.max(self.fitness) > global_best_fitness:
                            best_index = np.argsort(self.fitness)[-1]
                            global_best_fitness = self.fitness[best_index]
                            global_best_ind = self.pop[best_index, :]
                            count = 0
                        else:
                            count += 1
                        worst_index = np.argsort(self.fitness)[-1]
                        self.pop[worst_index, :] = global_best_ind
                        self.fitness[worst_index] = global_best_fitness

                    else:
                        if np.min(self.fitness) < global_best_fitness:
                            best_index = np.argsort(self.fitness)[0]
                            global_best_fitness = self.fitness[best_index]
                            global_best_ind = self.pop[best_index, :]
                            count = 0
                        else:
                            count += 1

                        worst_index = np.argsort(self.fitness)[-1]
                        self.pop[worst_index, :] = global_best_ind
                        self.fitness[worst_index] = global_best_fitness

                    if self.verbose != None and 0 == (it % self.verbose):
                        print('Gene {}:'.format(it))
                        print('Global best fitness:', global_best_fitness)

                    if self.early_stop_rounds != None and count > self.early_stop_rounds:
                        print('Did not improved within {} rounds. Break.'.format(self.early_stop_rounds))
                        break

                # print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_ind, global_best_fitness, eva_times))
                # print(global_best_ind)
                # print(global_best_fitness)
                return global_best_ind, global_best_fitness

        # *******************************************************************************************************************

        # *#*#*# Scenario 1 #*#*#*#
        print('Scenario 1: Maximum Accuracy with the Least Number of Predictors', file=f)
        # !pip install delayed
        features = np.zeros((Yield.shape[1], county_1_conditions.shape[1]))
        eff_lag_day = np.zeros((Yield.shape[1]))
        eff_lag_month = np.zeros((Yield.shape[1]))

        A = 1.3
        B = 1.2

        from sklearn.svm import SVR
        for ny in range(Yield.shape[1]):

            if ny == 0:
                Predictors = county_1_conditions
            else:
                Predictors = county_2_conditions

            global Max_tr, Max_te
            Max_tr = np.array(0.1)
            Max_te = np.array(0.1)

            # Apply effective lags to the predictors
            Lag_max = np.max(eff_lag[0, :, ny])

            Y = np.roll(Yield.iloc[:, ny].values, int(-Lag_max), axis=0)[:int(-Lag_max)]

            P_tot1 = np.zeros([Predictors.shape[0] - int(Lag_max), Predictors.shape[1]])
            for i in range(Predictors.shape[1]):
                P_tot1[:, i] = np.roll(Predictors.iloc[:, i].values, -int(Lag_max - eff_lag[0, i, ny]))[:-int(Lag_max)]

            P_tot = P_tot1  # Use environmental data as the only predictors

            # Split the data into train and test
            def values(arr):
                if np.sum(arr) < 1:
                    from random import randint
                    arr[randint(0, arr.shape[0] - 1)] = 1

                P = P_tot[:, [arr > 0][0]]

                # Random split
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(P, Y, test_size=0.33, random_state=42)

                # Data normalization
                def norm(x):
                    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

                normed_x_train = norm(x_train)
                normed_x_test = norm(x_test)

                # build SVR model  *****************************************

                model = SVR(C=20, epsilon=0.01, kernel='rbf',
                            degree=3, gamma='scale', coef0=0.0, tol=0.001,
                            shrinking=True, cache_size=200,
                            verbose=False, max_iter=-1)
                # model.summary()

                # SVR model training ******************************************
                history = model.fit(normed_x_train, y_train)

                train_predictions = model.predict(normed_x_train).flatten()
                r2_train = r2_score(y_train, train_predictions)
                mae_train = mean_absolute_error(y_train, train_predictions)
                mse_train = mean_squared_error(y_train, train_predictions)
                agg_err_train = ((np.sqrt(mse_train) + mae_train) / 2) * (1 - r2_train)

                test_predictions = model.predict(normed_x_test).flatten()
                r2_test = r2_score(y_test, test_predictions)
                mae_test = mean_absolute_error(y_test, test_predictions)
                mse_test = mean_squared_error(y_test, test_predictions)
                agg_err_test = ((np.sqrt(mse_test) + mae_test) / 2) * (1 - r2_test)

                global Max_tr, Max_te

                if r2_train > Max_tr:
                    Max_tr = r2_train
                    Max_te = r2_test

                Cost = abs(r2_train * A + B * r2_test + (
                            (np.mean([(Max_tr) * A, (Max_te) * B]) / 3 + 0.15) / (arr.shape[0] / 2.7)) * arr.shape[
                               0] / np.sum(arr))
                # Cost=abs(r2_train*A+B*r2_test)

                if r2_train < 0 or r2_test < 0:
                    Cost = 0

                # print(Cost, "\n")
                return Cost

            # Run feature selection model
            num_pop = 20
            problem_dimentions = P_tot.shape[1]

            test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, p_c=0.8, p_m=0.2, max_round=70,
                       early_stop_rounds=None, verbose=None, maximum=True)
            best_solution, best_fitness = test.run()
            features[ny, :] = best_solution
            eff_lag_day[ny] = round(np.mean(eff_lag[0, features[ny, :] > 0, ny]))
            eff_lag_month[ny] = round(np.mean(eff_lag[1, features[ny, :] > 0, ny]))
            print('\n FP_type: {} \n Best_features: {} \n Effective_daily_lag: {} \n Effective_monthly_lag: {}'.format(
                Yield.columns[ny], Predictors.columns[features[ny, :] > 0], eff_lag_day[ny], eff_lag_month[ny]), file=f)

        # Similarity percentage based on the Effective Perdictors and Lag-time
        P_lag1 = 100 - (abs(eff_lag_month[0] - eff_lag_month[1]) * 5)
        P_pred1 = 90 - (sum(abs(features[1, :] - features[0, :])) * 25)

        # Check train and test accuracy- Scenario 1
        R1 = np.zeros(2)
        for ny in range(Yield.shape[1]):
            # print(features[ny,:])

            if ny == 0:
                Predictors = county_1_conditions
            else:
                Predictors = county_2_conditions

            arr = features[ny, :]

            # Apply effective lags to the predictors
            Lag_max = eff_lag_day[ny].astype(np.int)

            Y = np.roll(Yield.iloc[:, ny].values, int(-Lag_max), axis=0)[:int(-Lag_max)]
            P_tot = Predictors.iloc[:-Lag_max, :].values

            # Split the data into train and test
            if np.sum(arr) < 1:
                from random import randint
                arr[randint(0, arr.shape[0] - 1)] = 1

            P = P_tot[:, [arr > 0][0]]

            # Random split
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(P, Y, test_size=0.33, random_state=42)

            def norm(x):
                return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

            normed_x_train = norm(x_train)
            normed_x_test = norm(x_test)

            # build SVR model *****************************************

            model = SVR(C=1000, epsilon=0.01, kernel='rbf',
                        degree=3, gamma='scale', coef0=0.0, tol=0.001,
                        shrinking=True, cache_size=200,
                        verbose=False, max_iter=-1)
            # model.summary()

            # SVR model training ******************************************
            history = model.fit(normed_x_train, y_train)

            train_predictions = model.predict(normed_x_train).flatten()
            r2_train = r2_score(y_train, train_predictions)
            mae_train = mean_absolute_error(y_train, train_predictions)
            mse_train = mean_squared_error(y_train, train_predictions)
            agg_err_train = ((np.sqrt(mse_train) + mae_train) / 2) * (1 - r2_train)

            test_predictions = model.predict(normed_x_test).flatten()
            r2_test = r2_score(y_test, test_predictions)
            mae_test = mean_absolute_error(y_test, test_predictions)
            mse_test = mean_squared_error(y_test, test_predictions)
            agg_err_test = ((np.sqrt(mse_test) + mae_test) / 2) * (1 - r2_test)

            R1[ny] = r2_train

            print("\n", Yield.columns[ny], file=f)
            print("R-square value on test data for Scenario 1: {:5.2f} ".format(r2_test), file=f)
            print("R-square value on train data for Scenario 1: {:5.2f} ".format(r2_train), file=f)

        # *#*#*# Scenario 2 #*#*#*#
        print('\n \n Scenario 2: Maximum Accuracy', file=f)
        # !pip install delayed
        features = np.zeros((Yield.shape[1], county_1_conditions.shape[1]))
        eff_lag_day = np.zeros((Yield.shape[1]))
        eff_lag_month = np.zeros((Yield.shape[1]))

        Max_tr = np.array(0.1)
        Max_te = np.array(0.1)
        A = 1.3
        B = 1.2

        from sklearn.svm import SVR

        for ny in range(Yield.shape[1]):

            if ny == 0:
                Predictors = county_1_conditions
            else:
                Predictors = county_2_conditions

            Max_tr = np.array(0.1)
            Max_te = np.array(0.1)

            # Apply effective lags to the predictors
            Lag_max = np.max(eff_lag[0, :, ny])

            Y = np.roll(Yield.iloc[:, ny].values, int(-Lag_max), axis=0)[:int(-Lag_max)]

            P_tot1 = np.zeros([Predictors.shape[0] - int(Lag_max), Predictors.shape[1]])
            for i in range(Predictors.shape[1]):
                P_tot1[:, i] = np.roll(Predictors.iloc[:, i].values, -int(Lag_max - eff_lag[0, i, ny]))[:-int(Lag_max)]

            # P_tot=np.append(P_tot1, Avg_Y, axis=1)
            P_tot = P_tot1

            # Split the data into train and test
            def values(arr):

                if np.sum(arr) < 1:
                    from random import randint
                    arr[randint(0, arr.shape[0] - 1)] = 1

                P = P_tot[:, [arr > 0][0]]

                # Random split
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(P, Y, test_size=0.33, random_state=42)

                def norm(x):
                    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

                normed_x_train = norm(x_train)
                normed_x_test = norm(x_test)

                # build SVR model  *****************************************

                model = SVR(C=1000, epsilon=0.01, kernel='rbf',
                            degree=3, gamma='scale', coef0=0.0, tol=0.001,
                            shrinking=True, cache_size=200,
                            verbose=False, max_iter=-1)
                # model.summary()

                # SVR model training ******************************************
                history = model.fit(normed_x_train, y_train)

                train_predictions = model.predict(normed_x_train).flatten()
                r2_train = r2_score(y_train, train_predictions)
                mae_train = mean_absolute_error(y_train, train_predictions)
                mse_train = mean_squared_error(y_train, train_predictions)
                agg_err_train = ((np.sqrt(mse_train) + mae_train) / 2) * (1 - r2_train)

                test_predictions = model.predict(normed_x_test).flatten()
                r2_test = r2_score(y_test, test_predictions)
                mae_test = mean_absolute_error(y_test, test_predictions)
                mse_test = mean_squared_error(y_test, test_predictions)
                agg_err_test = ((np.sqrt(mse_test) + mae_test) / 2) * (1 - r2_test)

                global Max_tr, Max_te

                if r2_train > Max_tr:
                    Max_tr = r2_train
                    Max_te = r2_test

                # Cost=abs(r2_train*A+B*r2_test+((np.mean([(Max_tr)*A,(Max_te)*B])/3+0.15)/(arr.shape[0]/2.7))*arr.shape[0]/np.sum(arr))
                Cost = abs(r2_train * A + B * r2_test)

                if r2_train < 0 or r2_test < 0:
                    Cost = 0

                return Cost

            # Run feature selection model
            num_pop = 20
            problem_dimentions = P_tot.shape[1]

            test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, p_c=0.8, p_m=0.2, max_round=70,
                       early_stop_rounds=None, verbose=None, maximum=True)
            best_solution, best_fitness = test.run()
            features[ny, :] = best_solution
            eff_lag_day[ny] = round(np.mean(eff_lag[0, features[ny, :] > 0, ny]))
            eff_lag_month[ny] = round(np.mean(eff_lag[1, features[ny, :] > 0, ny]))
            print('\n FP_type: {} \n Best_features: {} \n Effective_daily_lag: {} \n Effective_monthly_lag: {}'.format(
                Yield.columns[ny], Predictors.columns[features[ny, :] > 0], eff_lag_day[ny], eff_lag_month[ny]), file=f)

        # Similarity percentage based on the Effective Perdictors and Lag-time- Scenario 2
        P_lag2 = 100 - (abs(eff_lag_month[0] - eff_lag_month[1]) * 5)
        P_pred2 = 90 - (sum(abs(features[1, :] - features[0, :])) * 10)

        # Check train and test accuracy- Scenario 1
        R2 = np.zeros(2)
        for ny in range(Yield.shape[1]):
            # print(features[ny,:])

            if ny == 0:
                Predictors = county_1_conditions
            else:
                Predictors = county_2_conditions

            arr = features[ny, :]
            # Apply effective lags to the predictors
            Lag_max = eff_lag_day[ny].astype(np.int)
            # print(Lag_max)
            Y = np.roll(Yield.iloc[:, ny].values, int(-Lag_max), axis=0)[:int(-Lag_max)]
            P_tot = Predictors.iloc[:-Lag_max, :].values

            # P_tot=np.zeros([Predictors.shape[0]-int(Lag_max),Predictors.shape[1]])
            # for i in range(Predictors.shape[1]):
            # P_tot[:,i]=np.roll(Predictors.iloc[:,i].values,-int(Lag_max-eff_lag[0,i,ny]))[:-int(Lag_max)]

            # Split the data into train and test

            if np.sum(arr) < 1:
                from random import randint
                arr[randint(0, arr.shape[0] - 1)] = 1

            P = P_tot[:, [arr > 0][0]]

            ##1. for time series forecast
            # from sktime.forecasting.model_selection import temporal_train_test_split
            # from sktime.performance_metrics.forecasting import smape_loss
            # x_train,x_test=temporal_train_test_split(P, test_size=int(len(P)*0.2))
            # y_train,y_test=temporal_train_test_split(Y, test_size=int(len(P)*0.2))

            # Random split
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(P, Y, test_size=0.33, random_state=42)

            def norm(x):
                return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

            normed_x_train = norm(x_train)
            normed_x_test = norm(x_test)

            # build model 2= SVR *****************************************

            model = SVR(C=1000, epsilon=0.01, kernel='rbf',
                        degree=3, gamma='scale', coef0=0.0, tol=0.001,
                        shrinking=True, cache_size=200,
                        verbose=False, max_iter=-1)
            # model.summary()

            # SVR model training ******************************************
            history = model.fit(normed_x_train, y_train)

            train_predictions = model.predict(normed_x_train).flatten()
            r2_train = r2_score(y_train, train_predictions)
            mae_train = mean_absolute_error(y_train, train_predictions)
            mse_train = mean_squared_error(y_train, train_predictions)
            agg_err_train = ((np.sqrt(mse_train) + mae_train) / 2) * (1 - r2_train)

            test_predictions = model.predict(normed_x_test).flatten()
            r2_test = r2_score(y_test, test_predictions)
            mae_test = mean_absolute_error(y_test, test_predictions)
            mse_test = mean_squared_error(y_test, test_predictions)
            agg_err_test = ((np.sqrt(mse_test) + mae_test) / 2) * (1 - r2_test)

            R2[ny] = r2_train

            print("\n", Yield.columns[ny], file=f)
            print("R-square value on test data for Scenario 2: {:5.2f} ".format(r2_test), file=f)
            print("R-square value on train data for Scenario 2: {:5.2f} ".format(r2_train), file=f)

        # Final similarity percentage
        PS_Lag = np.mean([P_lag1, P_lag2])
        PS_Pred = np.int(np.mean([P_pred1, P_pred2]) * (1 - abs(np.mean([R1[0], R2[0]]) - np.mean([R1[1], R2[1]]))))

        return (PS_Lag, PS_Pred)

        # ********* Detail Analysis Output ************************

    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages("/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Inputs_Output_Plots.pdf")
    f = open('/Users/mohita/Documents/GitHub/VersatileFPApp/src/File/Similarity_Inputs_Output_Results.txt', 'w')

    interstage_dependency(county_1_conditions, county_2_conditions, Yield, pp)

    print('**Effective Lag-time of Predictors as a Result of the Interstage_Dependency_Analysis** \n', file=f)
    Prc[0] = Effective_lag(county_1_conditions, county_2_conditions, Yield, f)

    print('**Effective Predictors as a Result Feature Selection Approach** \n', file=f)
    Prc[1], Prc[2] = feature_selection(county_1_conditions, county_2_conditions, Yield, f)

    # Display
    from tabulate import tabulate
    my_df = pd.DataFrame(np.transpose(Prc), index=List_Prc, columns=['Percentage of Similarity'])
    print("\n", tabulate(my_df, headers='keys', tablefmt='psql'), file=f)

    Weighted_Prc = Prc
    Weighted_Prc[0] = Weighted_Prc[0] * 2
    Weighted_Prc[2] = Weighted_Prc[2] * 3
    percentage_similarity = round(np.sum(Weighted_Prc) / 6)

    print("\n Overal Similarity Percentage = {:5.2f} ".format(percentage_similarity), file=f)

    if percentage_similarity > 75:

        binary_similarity = '\n The investigated {} and {} are similar.'.format(Yield.columns[0], Yield.columns[1])
    else:
        binary_similarity = '\n The investigated {} and {} are dissimilar.'.format(Yield.columns[0], Yield.columns[1])

    print(binary_similarity, file=f)
    pp.close()
    f.close()

