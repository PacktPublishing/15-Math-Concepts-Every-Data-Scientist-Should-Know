# 15 Math Concepts Every Data Scientist Should Know

<a href="https://www.packtpub.com/en-in/product/15-math-concepts-every-data-scientist-should-know-9781837634187?type=print"><img src="https://content.packt.com/_/image/original/B19496/cover_image_large.jpg" alt="no-image" height="256px" align="right"></a>

This is the code repository for [15 Math Concepts Every Data Scientist Should Know](https://www.packtpub.com/en-in/product/15-math-concepts-every-data-scientist-should-know-9781837634187?type=print), published by Packt.

**Understand and learn how to apply the math behind data science algorithms**

## What is this book about?
As machine learning algorithms become more powerful, data scientists need a clear grasp of their key components. This book explains the core math principles underpinning the most used algorithms, detailing their importance and practical applications.

This book covers the following exciting features:
* Master foundational concepts that underpin all data science applications
* Use advanced techniques to elevate your data science proficiency
* Apply data science concepts to solve real-world data science challenges
* Implement the NumPy, SciPy, and scikit-learn concepts in Python
* Build predictive machine learning models with mathematical concepts
* Gain expertise in Bayesian non-parametric methods for advanced probabilistic modeling
* Acquire mathematical skills tailored for time-series and network data types

If you feel this book is for you, get your [copy](https://www.amazon.com/Math-Concepts-Every-Scientist-Should/dp/1837634181/ref=sr_1_1?crid=1C2WO9OBN7K7A&dib=eyJ2IjoiMSJ9.4JR7BfVPNYtsfHU1qm_lBDil6IpNIZFXgs7ocAKUu4LGjHj071QN20LucGBJIEps.Pguc7eQd77NmHrBgLS-BgqjwAYL5ZjxJf4e79kcrVXg&dib_tag=se&keywords=15+Math+Concepts+Every+Data+Scientist+Should+Know&qid=1721041833&sprefix=%2Caps%2C908&sr=8-1) today!
<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>
## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
map_estimate = minimize(neg_log_posterior,
                        x0,
                        method='BFGS',
                        options={'disp': True})
# Convert from logit(p) to p
p_optimal = np.exp(map_estimate['x'][0])/ (
    1.0 + np.exp(map_estimate['x'][0]))
print("MAP estimate of success probability = ", p_optimal)
```

**Following is what you need for this book:**
This book is for data scientists, machine learning engineers, and data analysts who already use data science tools and libraries but want to learn more about the underlying math. Whether you’re looking to build upon the math you already know, or need insights into when and how to adopt tools and libraries to your data science problem, this book is for you. Organized into essential, general, and selected concepts, this book is for both practitioners just starting out on their data science journey and experienced data scientists.

With the following software and hardware list you can run all code files present in the book (Chapter 2-15).
## Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 2-15 | Python, Jupyter Notebook | Windows, macOS, or Linux |

## Related products
* Principles of Data Science - Third Edition [[Packt]](https://www.packtpub.com/en-in/product/principles-of-data-science-9781837636303?type=print) [[Amazon]](https://www.amazon.com/Principles-Data-Science-beginners-essential/dp/1837636303/ref=sr_1_1?crid=35TB94CZRI167&dib=eyJ2IjoiMSJ9.H85FZKe8pNLP2vQLnpr2kYCh4DCCjJMv_tSP5ytPaoczmN46fPTb8PfiB0HstqQnAUUEZtsSCmJ1wy3hAZ6dsrxW0kbDGLaYRZ-M_ivtMsgpkOyd_3CREnug13vGgVLnAl1i5c6-S7gc0OyO0FArtHSqGoNs30PgVIeOpsUXTULnmqbcRZz-KG7mBsvkcIWYWj4tRKZwVD1BDSCilEjfgqbY8N8KnO75J0LyhhlUCVM.Eh2dM80mzKjCy5BzPDi6LtwhwNGm-HQF4AiJXWOQcq8&dib_tag=se&keywords=Principles+of+Data+Science&qid=1721042571&sprefix=principles+of+data+science%2Caps%2C881&sr=8-1)

* Cracking the Data Science Interview [[Packt]](https://www.packtpub.com/en-in/product/cracking-the-data-science-interview-9781805120506?type=print) [[Amazon]](https://www.amazon.com/Cracking-Data-Science-Interview-industry/dp/1805120506/ref=sr_1_4?crid=1SEDX8L5N26KM&dib=eyJ2IjoiMSJ9.MHG7FrZsxVUx3I6fbbfvwQEaVUaydi5Uukh921X5WC3fHl56hDG8oRAh-ZQuLhzJwmNOsaeObb06azSZyCOEDygJwkBiuUGMFp51Op7vlI1g9jc1vWO42Gpz4nop0MKAw8nflLetXc1dRTV_hC5NH6DiFIIN0j0pgSQEgUZWONLLW0l40dTybNKoVjl2Ci0gRBxPSqGFypXl33uN3XayKWYvUFr_8qMCSpgdB_4EuSw.7ntJHtbCbBi5UtUvoQAyOpreQbh1SzmCAxqj_NweqsA&dib_tag=se&keywords=Cracking+the+Data+Science+Interview&qid=1721042647&sprefix=cracking+the+data+science+interview%2Caps%2C822&sr=8-4)

## Get to Know the Author
**David Hoyle**
has over 30 years&rsquo; experience in statistical and mathematical modelling. After gaining a degree in Mathematics and Physics and a PhD in Theoretical Physics from the University of Bristol in the UK, he began an academic career that included research at the University of Cambridge and leading his own machine learning research groups at the University of Exeter and the University of Manchester in the UK.
Since 2011 he has worked as a Data Scientist in the private sector, including for Lloyds Banking Group, and AutoTrader UK as Head of Data Science. In 2019 he joined the customer data science company dunnhumby as a Lead Data Scientist, building statistical and machine learning predictive models for the world's largest retailers.


