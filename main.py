import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm, uniform, expon, poisson, bernoulli
from torch.distributions import Beta, Gamma
import torch
import math

st.set_page_config(page_title="Bayesian Inference", page_icon=":chart_with_upwards_trend:", layout="wide")
st.markdown('''
<style>
.katex-html {
    text-align: left;
}
</style>''',
unsafe_allow_html=True
)

def get_prior_params(prior):
    if prior == "Beta":
        param_a = st.slider("Enter Beta Parameter a", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
        param_b = st.slider("Enter Beta Parameter b", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
        return [param_a, param_b]
    elif prior == "Gamma":
        param_shape = st.slider("Enter Gamma Shape Parameter", min_value=0.01, max_value=10.0, value=1.0)
        param_scale = st.slider("Enter Gamma Scale Parameter", min_value=0.01, max_value=10.0, value=1.0)
        return [param_shape, param_scale]
    elif prior == "Normal":
        param_mean = st.slider("Enter Normal Mean Parameter", min_value=-10, max_value=10, value=0, step=1)
        param_std = st.slider("Enter Normal Standard Deviation Parameter", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
        return [param_mean, param_std]
    elif prior == "Uniform":
        # No parameters needed for uniform
        return []
    elif prior == "Exponential":
        param_scale = st.slider("Enter Exponential Scale Parameter", min_value=0.01, max_value=10.0, value=1.0)
        return [param_scale]

def get_likelihood_params(likelihood):
    # Add more cases for different likelihood distributions
    if likelihood == "Bernoulli":
        param_p = st.slider("Enter Bernoulli Parameter p", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        return [param_p]
    elif likelihood == "Poisson":
        param_lambda = st.slider("Enter Poisson Parameter lambda", min_value=0.01, max_value=10.0, value=1.0)
        return [param_lambda]
    elif likelihood == "Normal":
        param_mean = st.slider("Enter Normal Mean Parameter", min_value=-10, max_value=10, value=0, step=1, key="slider1")
        param_std = st.slider("Enter Normal Standard Deviation Parameter", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="slider2")
        return [param_mean, param_std]

def conjugate_prior_info(prior, likelihood):
    if prior == "Beta" and likelihood == "Bernoulli":
        return "No transformation needed", "Beta (Conjugate)"
    elif prior == "Gamma" and likelihood == "Poisson":
        return "No transformation needed", "Gamma (Conjugate)"
    elif prior == "Normal" and likelihood == "Normal":
        return "No transformation needed", "Normal (Conjugate)"
    else:
        return "Unknown", "Unknown (Non-Conjugate)"

def non_conjugate_prior_info(prior, likelihood):
    if prior == "Normal" and likelihood == "Bernoulli":
        return "Apply Sigmoid function to the parameter", "Unknown (Non-Conjugate)"
    elif prior == "Uniform" and likelihood == "Poisson":
        return "No transformation needed", "Unknown (Non-Conjugate)"
    elif prior == "Exponential" and likelihood == "Normal":
        return "No transformation needed", "Unknown (Non-Conjugate)"
    else:
        return "Unknown", "Unknown (Non-Conjugate)"

def plot_prior_posterior(prior, likelihood, prior_params, likelihood_params):
    pdf_space = np.linspace(-5,5,1000)
    
    if prior == "Beta":
        prior_dist = Beta(torch.tensor([prior_params[0]]), torch.tensor([prior_params[1]]))
        p_values = torch.linspace(0.001, 0.999, 10000)
        prior_values = prior_dist.log_prob(p_values).exp()
        prior_values = prior_values.numpy()
        #prior_dist = beta.pdf(pdf_space, prior_params[0], prior_params[1])
    elif prior == "Gamma":
        prior_dist = Gamma(torch.tensor([prior_params[0]]), torch.tensor([prior_params[1]]))
        p_values = torch.linspace(0, 10, 10000)
        prior_values = prior_dist.log_prob(p_values).exp()
        prior_values = prior_values.numpy()
    elif prior == "Normal":
        p_values = np.linspace(-5, 5, 10000)
        prior_values = norm.pdf(p_values, loc=prior_params[0], scale=prior_params[1])
    elif prior == "Uniform":
        prior_dist = uniform.pdf(pdf_space)
    elif prior == "Exponential":
        prior_dist = expon.pdf(pdf_space, scale=prior_params[0])

    if likelihood == "Bernoulli":
        n1 = int(likelihood_params[0]*100)
        n0 = int(100-n1)
        gc = math.gcd(n0,n1)
        n1 /= gc
        n0 /= gc
        likelihood_values = [(p ** n1) * ((1-p)**n0) for p in p_values]
        max_likelihood = max(likelihood_values)
    elif likelihood == "Poisson":
        likelihood_values = poisson.pmf(p_values, likelihood_params[0])
        max_likelihood = max(likelihood_values)
    elif likelihood == "Normal":
        likelihood_values = norm.pdf(likelihood_params[0], loc=p_values, scale=likelihood_params[1])
        max_likelihood = max(likelihood_values)

    # Update prior based on observed data (posterior distribution)
    if prior == "Beta" and likelihood == "Bernoulli":
        posterior_dist = Beta(torch.tensor([prior_params[0]+n1]), torch.tensor([prior_params[1]+n0]))
        posterior_values = posterior_dist.log_prob(p_values).exp()
        max_post = max(posterior_values)

        normalised_likelihood_values = []
        for i in likelihood_values:
            normalised_likelihood_values.append(i / max_likelihood * max_post)
        posterior_values = posterior_values.numpy()
        param_values = p_values.numpy()
    elif prior == "Gamma" and likelihood == "Poisson":
        posterior_dist = gamma.pdf(data, prior_params[0] + (likelihood_dist==1).sum(), scale=1/(prior_params[1]+num_likelihood_samples))
    elif prior == "Normal" and likelihood == "Normal":
        posterior_unnormalized = prior_values * likelihood_values
        posterior_values = posterior_unnormalized / np.sum(posterior_unnormalized) / (mu_values[1] - mu_values[0])
        max_post = max(posterior_values)

        normalised_likelihood_values = []
        for i in likelihood_values:
            normalised_likelihood_values.append(i / max_likelihood * max_post)
        param_values = p_values
    else:
        posterior_unnormalized = prior_values * likelihood_values
        posterior_values = posterior_unnormalized / np.sum(posterior_unnormalized) / (p_values[1] - p_values[0])
        max_post = max(posterior_values)

        normalised_likelihood_values = []
        for i in likelihood_values:
            normalised_likelihood_values.append(i / max_likelihood * max_post)
        param_values = p_values

    # Plotting
    fig, ax = plt.subplots()
    
    #ax[0].plot(p_values.numpy(), prior_values.numpy(), label='Prior')
    #ax[0].set_title('Prior Distribution')
    #ax[1].plot(p_values.numpy(), normalised_likelihood_values, label='Likelihood')
    #ax[1].set_title('Likelihood Distribution')
    #ax[2].plot(p_values.numpy(), posterior_values.numpy(), label='Posterior')
    #ax[2].set_title('Posterior Distribution')
    
    ax.plot(param_values, prior_values, label='Prior')
    ax.plot(param_values, normalised_likelihood_values, label='Likelihood')
    ax.plot(param_values, posterior_values, label='Posterior', linestyle='-.', color='black')
    ax.legend()

#    for a in ax:
        #a.legend()
#        a.grid(True)

    st.pyplot(fig)

def main():
    #st.title("Conjugate and Non-Conjugate Priors Information with Visualizations")
    col1, col2 = st.columns([1,2],gap="large")
    
    with col1:
        st.header("Prior Distribution")
        prior = st.selectbox("Select Prior Distribution", ["Beta", "Gamma", "Normal", "Exponential"])
        prior_params = get_prior_params(prior)

        st.divider()

        st.header("Likelihood Distribution")
        likelihood = st.selectbox("Select Likelihood Distribution", ["Bernoulli", "Normal", "Poisson"])
        likelihood_params = get_likelihood_params(likelihood)
    
    # Display information based on choices
    if prior in ["Beta", "Gamma", "Normal"] and likelihood in ["Bernoulli", "Poisson", "Normal"]:
        bijector, posterior_type = conjugate_prior_info(prior, likelihood)
    else:
        bijector, posterior_type = non_conjugate_prior_info(prior, likelihood)

    with col2:
        st.header("Posterior Distribution")
        # Display results
        #st.subheader("Results:")
        coll1, coll2 = st.columns([2,1])
        with coll2:
            st.write(f"Bijector (Parameter Transform): {bijector}")
            st.write(f"Posterior Distribution Type: {posterior_type}")

        # Visualizations
        #st.subheader("Visualizations:")
        with coll1:
            plot_prior_posterior(prior, likelihood, prior_params, likelihood_params)
        
        if prior == "Beta" and likelihood == "Bernoulli":
            st.latex(r'\textbf{Prior}\text{: Beta}(\alpha,\beta) \equiv \text{Beta}('+rf'{prior_params[0]}'+r','+rf'{prior_params[1]}'+r')')
            st.latex(r'\textbf{Likelihood}\text{: Bernoulli}(p) \equiv \text{Bernoulli}('+rf'{likelihood_params[0]}'+r')')
            st.latex(r"\textbf{Posterior}\text{: Beta}(\alpha',\beta') \equiv \text{Beta}\left(\alpha + \sum_{i=1}^{n} x_i,\beta + n - \sum_{i=1}^{n} x_i\right)")
        
        #if prior == "Normal" and likelihood == "Normal":
        #    st.latex(r'\textbf{Prior}\text{: Normal}(\mu_0,\sigma_0^2) \equiv \text{Normal}('+rf'{prior_params[0]}'+r','+rf'{prior_params[1]}^2'+r')')
        #    st.latex(r'\textbf{Likelihood}\text{: Normal}(\mu,\sigma^2) \equiv \text{Normal}('+rf'{likelihood_params[0]}'+r','+rf'{likelihood_params[1]}^2'+r')')
        #    st.latex(r"\textbf{Posterior}\text{: Normal}(\mu',\sigma'^2) \equiv \text{Normal}\left(\frac{1}{\frac{1}{\sigma_0^2}+\frac{n}{\sigma^2}}\right)")
            
if __name__ == "__main__":
    main()