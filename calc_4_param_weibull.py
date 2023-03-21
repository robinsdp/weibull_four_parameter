import scipy
from scipy.optimize import minimize
import numpy as np

class weibull_4_parameter:
    
    def __init__(self, values_to_fit = None):
        self.values_to_fit = values_to_fit
        self.B = None
        self.D = None
        self.m = None
        self.n = None
    
    # values_to_fit should be a numeric list of length > 4
    def set_values_to_fit(self, values_to_fit):
        self.values_to_fit = values_to_fit
    
    # Name & location of file to read it.
    # Data must be one number per line with no extra characters
    def load_values_to_fit(self, file_name_location):
        # Open up file to be read
        data_file = open(file_name_location, "r")
        
        file_contents = data_file.read()
        file_contents_to_list = file_contents.split('\n')
        
        def string_to_float(value):
            try:
                float_return = float(value)
                return(float_return)
            except ValueError:
                return(None)
        
        observed_values = [string_to_float(x) for x in file_contents_to_list]
        # Filters out the None values added from string_to_float function
        observed_values = [v for v in observed_values if v is not None]
        
        data_file.close()
        
        self.values_to_fit = observed_values
    
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
    def predict_4_param_weibul_cdf(self, B, D, n, m, x):
        predictions = [1 - (np.exp(-1/((n/v)**B + (m/v)**D))) for v in x]
        return predictions
    
    def export_4_param_weibul_cdf_double_log_preds(self, output_location, x):
        if self.B is None or self.D is None or self.m is None or self.n is None:
            return 'Unfilled parameter for weibull'
        def neg_double_log(t):
            return [np.log(-np.log(1-v)) for v in t]
        
        plot_preds_string = list(map(str, neg_double_log(self.predict_4_param_weibul_cdf(self.B, self.D, self.m, self.n, x))))
        plot_preds_string = ["{}\n".format(i) for i in plot_preds_string]
        with open("/home/davis.robinson/weibull_pred_values.txt", 'w') as fp:
            fp.writelines(plot_preds_string)
    
weib_1 = weibull_4_parameter()
weib_1.load_values_to_fit("/home/davis.robinson/weibull_x_values.txt")
mle1 = weib_1.calc_ideal_parameters()

incrementor = .05
vals_to_predict_no_power = [round(x*incrementor,3) for x in range(int(-3/incrementor),int(2.5/incrementor))]
plot_x_values_to_predict = [10**x for x in vals_to_predict_no_power]

predictions = weib_1.predict_4_param_weibul_cdf(weib_1.B, weib_1.D, weib_1.m, weib_1.n, plot_x_values_to_predict)
weib_1.export_4_param_weibul_cdf_double_log_preds("/home/davis.robinson/weibull_pred_values.txt", plot_x_values_to_predict)