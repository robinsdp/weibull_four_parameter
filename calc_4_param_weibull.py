import scipy
import argparse
from scipy.optimize import minimize
import numpy as np

class weibull_4_parameter:
    
    def __init__(self, values_to_fit = None):
        self.values_to_fit = values_to_fit
        self.B = None
        self.D = None
        self.m = None
        self.n = None

    # just displays the weibull distribution parameters
    def print_parameters(self):
        print('B: ', self.B)
        print('D: ', self.D)
        print('m: ', self.m)
        print('n: ', self.n)

    # Converts a list of values to the negative double log
    def neg_double_log(self, t):
        return [np.log(-np.log(1 - v)) for v in t]

    # values_to_fit should be a numeric list of length > 4
    def set_values_to_fit(self, values_to_fit):
        self.values_to_fit = values_to_fit

    # Load a file into a list
    def load_values(self, input_location):
        # Open up file to be read
        data_file = open(input_location, "r")

        file_contents = data_file.read()
        file_contents_to_list = file_contents.split('\n')

        def string_to_float(value):
            try:
                float_return = float(value)
                return (float_return)
            except ValueError:
                return (None)

        observed_values = [string_to_float(x) for x in file_contents_to_list]
        # Filters out the None values added from string_to_float function
        observed_values = [v for v in observed_values if v is not None]

        data_file.close()

        return observed_values

    # Name & location of file to read it.
    # Data must be one number per line with no extra characters
    def load_values_to_fit(self, input_location):
        self.values_to_fit = self.load_values(input_location)
    
    # # Returns the negative sum of logged probabilities from a given 4-param-Weibull distribution given its parameters and x values
    # def calc_4_param_weibul_neg_log_prob(self, B, D, n, m, x):
    #     # This next step calculates the probability of each observation from the distribution's PDF
    #     # Next each value is logged, and then the values are summed and multiplied by -1 
    #     # The log is only taken to avoid multiplication so the values don't explode when a large number of observations are present (or implode... near hitting the maximum precision of floats)
    #     neg_log_probability_product = -1 * sum(self.predict_4_param_weibul(B, D, n, m, x))
    #     return neg_log_probability_product
    
    # Returns a list of probability from a given 4 parameter weibull distribution given a list of x values.
    def predict_4_param_weibul(self, B, D, n, m, x):
        predictions = [(np.exp(-1/((m/v)**D + (n/v)**B))*(D*(m/v)**D + B*(n/v)**B))/(v*((m/v)**D + (n/v)**B)**2) for v in x]
        return predictions
    
    # Helper function for the Scipy minimize function
    # Uses the logged negative sum of probabilities to find the MLE
    def weibull_minimize_func(self, parameters_list = np.array([.1,.5,.2,.7])):
        B, D, n, m = parameters_list
        # -1 is to minimize instead of maximize
        # Minimizing the sum of the logged probabilities has the same outcome as the product of pdf probabilities. However, it results in less extreme numbers
        # meaning that the floating point precision limitations are not hit.
        LL_product = -1 * sum([np.log(prediction) for prediction in self.predict_4_param_weibul(B, D, n, m, self.values_to_fit)])
        return LL_product
    
    # Perform a search over the parameter space to find the one that finds the MLE
    def calc_ideal_parameters(self):
        mle_model = minimize(self.weibull_minimize_func, np.array([.3,.4,.1,.9]), bounds = ((0.000001, None), (0.000001, None), (0.000001, None), (0.000001, None)))
        self.B = mle_model.x[0]
        self.D = mle_model.x[1]
        self.m = mle_model.x[2]
        self.n = mle_model.x[3]
        return mle_model
    
    # Returns the cdf predictions from given weibutl model
    def predict_4_param_weibul_cdf(self, x, B = None, D = None, n = None, m = None, negative_double_log = True):
        if B is None:
            B = self.B
        if D is None:
            D = self.D
        if n is None:
            n = self.n
        if m is None:
            m = self.m
        predictions = [1 - (np.exp(-1/((n/v)**B + (m/v)**D))) for v in x]

        if negative_double_log:
            predictions = self.neg_double_log(predictions)

        return predictions

    # Exports the predictions to a text file, with each row being 1 prediction
    def export_4_param_weibul_cdf_preds(self, output_location, x, negative_double_log = True):
        if self.B is None or self.D is None or self.m is None or self.n is None:
            return 'Unfilled parameter for weibull'
        
        plot_preds_string = list(map(str, self.predict_4_param_weibul_cdf(x, negative_double_log = negative_double_log)))
        plot_preds_string = ["{}\n".format(i) for i in plot_preds_string]
        with open(output_location, 'w') as fp:
            fp.writelines(plot_preds_string)

# Fits the 4 parameters, fits on another file of values, and exports predictions to a third file
def fit_and_export(input_location, output_location = None, prediction_values = None, negative_double_log = True):
    weib_1 = weibull_4_parameter()
    weib_1.load_values_to_fit(input_location)
    mle1 = weib_1.calc_ideal_parameters()

    weib_1.print_parameters()

    if output_location is not None and prediction_values is not None:
        values_to_fit = weib_1.load_values(prediction_values)
        weib_1.export_4_param_weibul_cdf_preds(output_location, values_to_fit, negative_double_log = negative_double_log)


def main():
    """
    main
    """
    parser = argparse.ArgumentParser(description='Weibull 4 Parameter Distribution Fit')
    # use this flag, or change the default here to use different graph description files
    parser.add_argument('-i', '--inputFile', help='File with values to fit to distribution', default=None, dest='inputFile')
    parser.add_argument('-l', '--negativeDoubleLog', help='Compute the Negative Double Log on predictions?', default=True, dest='negativeDoubleLog')
    parser.add_argument('-o', '--outputFile', help='Output File Location. Requires a prediction file as well', default=None, dest='outputFile')
    parser.add_argument('-x', '--predictionFile', help='File of values on which to form predictions from the resulting distribution', default=None, dest='predictionFile')
    args = parser.parse_args()

    fit_and_export(input_location=args.inputFile,
                   output_location=args.outputFile,
                   prediction_values=args.predictionFile,
                   negative_double_log=args.negativeDoubleLog)


if __name__ == '__main__':
    main()
