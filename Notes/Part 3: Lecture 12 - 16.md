Lecture 12 -- Network Effects and Cascading Behavior
===================================================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/12-cascades.pdf), [Video](https://youtu.be/nNiRSQN9-ts)*

Information can spread through networks: behaviors that cascade from node to node like an epidemic:

-   Cascading behavior.
-   Diffusion of innovations.
-   Network effects.
-   Epidemics.

Examples of spreading through networks: 

-   Biological: diseases via contagion.
-   Technological: cascading failures, spread of information.
-   Social: rumors, news, new technology, viral marketing.

Network cascades are contagions that spread over the edges of the network. It creates a propagation tree, i.e., cascade. "Infection" event in this case can be adoption, infection, activation.

Two ways to model model diffusion:

-   Decision based models (this lecture ): 
    -   Models of product adoption, decision making.
    -   A node observes decisions of its neighbors and makes its own decision.
    -   Example: you join demonstrations if *k* of your friends do so.
-   Probabilistic models (next lecture): 
    -   Models of influence or disease spreading.
    -   An infected node tries to "push" the contagion to an uninfected node.
    -   Example: you "catch" a disease with some probability from each active neighbor in the network.

###### * Decision based diffusion models*

Game Theoretic model of cascades:

Based on 2 player coordination game: 2 players -- each chooses technology A or B; each player can only adopt one "behavior", A or B; intuition is such that node *v* gains more payoff if *v*'s friends have adopted the same behavior as *v.*

Payoff matrix for model with two nodes looks as following:

-   If both *v* and *w* adopt behavior *A*, they each get payoff *a > 0.*
-   If *v* and *w* adopt behavior *B*, they each get payoff *b > 0.*
-   If *v* and *w* adopt the opposite behaviors, they each get 0.
-   In some large network: each node *v* is playing a copy of the game with each of its neighbors. Payoff is the sum of node payoffs over all games.

Calculation of node *v*:

-   Let *v* have *d* neighbors. 
-   Assume fraction *p* of *v*'s neighbors adopt *A*.
-   *Payoff**v** = a∙p∙d* if *v* chooses *A; Payoff**v** = b∙(1-p)∙d* if v chooses *B.*
-   Thus: *v* chooses *A* if  *p > b / (a+b) = q* or  *p > q *(*q* is payoff threshold)*.*

*Example Scenario:*

Assume graph where everyone starts with all B.  Let small set S of early adopters of A be hard-wired set S -- they keep using A no matter what payoffs tell them to do. Assume payoffs are set in such a way that nodes say: if more than q = 50% of my friends take A, I'll also take A. This means: *a = b -- ε *(*ε>0*, small positive constant) and then* q = ½*.

###### *Application: modeling protest recruitment on social networks*

Case: during anti-austerity protests in Spain in May 2011, Twitter was used to organize and mobilize users to participate in the protest. Researchers identified 70 hashtags that were used by the protesters.

-   70 hashtags were crawled for 1 month period. Number of tweets: 581,750.
-   Relevant users: any user who tweeted any relevant hashtag and their followers + followees. Number of users: 87,569.
-   Created two undirected follower networks:

1\. Full network: with all Twitter follow links.

2\. Symmetric network with only the reciprocal follow links (*i ➞ j *and *j ➞ i*). This network represents "strong" connections only.

Definitions:

-   User activation time: moment when user starts tweeting protest messages.
-   *kin* = the total number of neighbors when a user became active.
-   *ka* = number of active neighbors when a user became active.
-   Activation threshold = *ka/kin*. The fraction of active neighbors at the time when a user becomes active.
    -   If *ka/kin ≈ 0*, then the user joins the movement when very few neighbors are active ⇒ no social pressure.
    -   If *ka/kin ≈ 1*, then the user joins the movement after most of its neighbors are active ⇒ high social pressure.

![](https://lh4.googleusercontent.com/ORLnj5xbPdLJ2YWAJ-BACysw6A8b96KBs4JvE9jixva688Fe4sJjaHIqIF7RRtZ3nsSxoWiSSiZ1t6hI2X2QfdH7yWVtik1KzonBpi_wuJeOHX3Y6fTveUquWlKKBejnIgBZ_tZF)

*Distribution of activation thresholds -- mostly uniform distribution in both networks, except for two local peaks.*

Authors define the information cascades as follows: if a user tweets a message at time *t* and one of its followers tweets a message in *(t, t + Δt)*, then they form a cascade.

![](https://lh3.googleusercontent.com/Sr5AJEXNaFsxIHsq5YOuI5RUzy6gvamXRy_t-G_YH9cMiMmtPM3CIOHWHykknhAWcf1c0S6NHvEjAcMItladf1X7i-uPgjPVVZNgUEED8r-kkAG3bfaBxFUJAfhdWjOn83P897FN)

*Cascade formed from 1 ➞ 2 ➞ 3*

To identify who starts successful cascades, authors use method of k-core decomposition: 

-   k-core: biggest connected subgraph where every node has at least degree k. 
-   Method: repeatedly remove all nodes with degree less than k.
-   Higher k-core number of a node means it is more central.

Picture below shows the K-core decomposition of the follow network: red nodes start successful cascades and they have higher k-core values. So, successful cascade starters are central and connected to equally well connected users.

![](https://lh5.googleusercontent.com/JE73qvl4Sfivt4huJ77vteADzWQyqkR0CmKLxLWTX04hfoGiW1tdipALsARoK0CbPRsAslxlKmC_vLcrBnfVA41pSfBJl4iQ_-4H-VuxK_616150YhwyhTQYPaC0w3Dx8OPGZfjm)

*K-core decomposition of the follow network*

To summarize the cascades on Twitter: users have uniform activation threshold, with two local peaks, most cascades are short, and successful cascades are started by central (more core) users.

So far, we looked at Decision Based Models that are utility based, deterministic and "node" centric (a node observes decisions of its neighbors and makes its own decision -- behaviors A and B compete, nodes can only get utility from neighbors of same behavior: A-A get *a*, B-B get *b*, A-B get *0*). The next model of cascading behavior is the extending decision based models to multiple contagions.

###### *Extending the Model: Allow people to adopt A and B*

Let's add an extra strategy "AB":

-   AB-A : gets *a*.
-   AB-B : gets *b*.
-   AB-AB : gets *max(a, b)*.
-   Also: some cost *c* for the effort of maintaining both strategies (summed over all interactions).
    -   Note: a given node can receive *a* from one neighbor and *b* from another by playing AB, which is why it could be worth the cost *c.*

Cascades and compatibility model: every node in an infinite network starts with B. Then, a finite set* S* initially adopts A. Run the model for *t=1,2,3,...*: each node selects behavior that will optimize payoff (given what its neighbors did in at time *t-1*). How will nodes switch from B to A or AB?

Let's solve the model in a general case: infinite path starts with all Bs. Payoffs for *w*: *A = a, B = 1, AB = a+1-c.* For what pairs *(c,a)* does A spread? We need to analyze two cases for node *w (*different neighbors situations) and define what *w *would do based on the values of *a* and *c*?

1.  A-w-B:

![](https://lh3.googleusercontent.com/-KOiSpK3gAXXXTkBC6mvqG6AlCMBDRblPj-GUzFyO3QecMakTLXE8ep-kfIwT70-NKnppRtUpxQZcnAY_ThSQ20H-jew5bg5BqT2UrlzfBA478xHnoLNI1KIzw6afz3YpcRAxSNv)

*Color letters indicate what is the optimal decision for node w (with payoffs for w: A = a, B = 1, AB = a+1-c).*

1.  AB-w-B:

![](https://lh6.googleusercontent.com/hUYgwBOvCXNfjKUa6k8pzipVvXTx2qt0Ne4ZaU1GVTVJ7kXb5cmVn39N1ipqOixKZiwvqyaI7t0g1se-MHZSLSxu2OxMSt5bUbCPfQ6wYhOyiJJD1CaET3p739U9aZdkYYpfvJB4)

*Color letters indicate what is the optimal decision for node w (now payoff is changed for B: A = a, B = 1+1, AB = a+1-c).*

If we combine two pictures, we get the following:

![](https://lh5.googleusercontent.com/lKbth8Gu0QFpn2LvQp1jP59kY1PsBTpZTcHRTtRAkeq9r90r90bH8lJ8lUn3gdSaTdQhoVwvaemuTiniwvfeMrW9XRS0-xEGJKRkRNg6IpEQeuLlMT4mOS6vArrUmjCdro8Gfmx7)

To summarise: if B is the default throughout the network until new/better A comes along, what happens:

-   Infiltration: if B is too compatible then people will take on both and then drop the worse one (B).
-   Direct conquest: if A makes itself not compatible -- people on the border must choose. They pick the better one (A).
-   Buffer zone: if you choose an optimal level then you keep a static "buffer" between A and B.

* * * * *

Lecture 13 -- Probabilistic Contagion and Models of Influence
============================================================

[*Slides*](http://snap.stanford.edu/class/cs224w-2019/slides/13-contagion.pdf)*, *[*Video*](https://youtu.be/dKkJJX_sf6c)

So far, we learned deterministic decision-based models where nodes make decisions based on pay-off benefits of adopting one strategy or the other. Now, we will do things by observing data because in cascades spreading like epidemics, there is lack of decision making and the process of contagion is complex and unobservable (in some cases it involves (or can be modeled as) randomness).

Simple model: Branching process

-   First wave: a person carrying a disease enters the population and transmits to all she meets with probability *q*. She meets *d* people, a portion of which will be infected. 
-   Second wave: each of the *d* people goes and meets *d* different people. So we have a second wave of *d* ∗ *d* = *d2* people, a portion of which will be infected. 
-   Subsequent waves: same process.

Epidemic model based on random trees

-   A patient meets *d* new people and with probability *q>0* she infects each of them.
-   Epidemic runs forever if: *lim (h→∞) p(h) > 0* (*p(h)* is probability that a node at depth h is infected.
-   Epidemic dies out if:  *lim (h→∞) p(h) = 0.*

So, we need to find  *lim (h→∞) p(h)* based on *q* and *d*. For *p(h) *to be recurrent (parent-child relation in a tree):

![ p_h = 1-(1-q\cdot p_{h-1})^d ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-403ab2ed10aead75c2bd2f9cbc83758e_l3.svg "Rendered by QuickLaTeX.com")

Then *lim (h→∞) p(h) *equals the result of iterating *f(x) = 1 -- (1 -- q ⋅ x)**d*where* x**1** = f(1) = 1 (since p**1 **= 1), **x**2** = f(**x**1**),  **x**3** = f(**x**2**), ...*

If we want the epidemic to die out, then iterating *f(x)* must go to zero. So, *f(x)* must be below* y = x.*

![](https://lh6.googleusercontent.com/vkVMGuUMUbKALoLUlZ8mtGLjWrJeqABYaHB1B2sFbGZO0y1l5xQnVfKQ5VHF1db03CNjS3wlHecK_OimqVDVFWYyJMq9TFRcmwBagwMhE_fYNg0wpt-2Ya0H83mOaVx8I2lqT-By)

The shape of *f(x) *is monotone.

Also, *f′(x)*  is non-decreasing, and * f′(0) = d ⋅ q*, which means that for *d ⋅ q > 1*, the curve is above the line, and we have a single fixed point of* x = 1* (single because *f′* is not just monotone, but also non-increasing). Otherwise (if *d ⋅ q > 1*), we have a single fixed point *x = 0*. So a simple result: depending on how *d ⋅ q* compares to 1, epidemic spreads or dies out.

Now we come to the most important number for epidemic *R0 = d ⋅ q *(*d ⋅ q* is expected # of people that get infected). There is an epidemic if *R0 ≥  1.*

Only *R**0* matters:

-   *R**0* ≥ 1: epidemic never dies and the number of infected people increases exponentially.
-   *R**0*  < 1: Epidemic dies out exponentially quickly.

Measures to limit the spreading: When *R0* is close 1, slightly changing *q* or *d* can result in epidemics dying out or happening:

-   Quarantining people/nodes [reducing *d*].
-   Encouraging better sanitary practices reduces germs spreading [reducing *q*].
-   HIV has an *R0* between 2 and 5.
-   Measles has an *R0* between 12 and 18.
-   Ebola has an *R0* between 1.5 and 2.

###### *Application: Social cascades on Flikr and estimating R**0** from real data*

*The paper *[*Characterizing social cascades in Flickr*](http://www.ccs.neu.edu/home/amislove/publications/Cascades-WOSN.pdf)* [Cha et al. ACM WOSN 2008] has all details about the application.*

Dataset:

-   Flickr social network: users are connected to other users via friend links; a user can "like/favorite" a photo.
-   Data: 100 days of photo likes; 2 million of users; 34,734,221 likes on 11,267,320 photos.

Cascades on Flickr:

-   Users can be exposed to a photo via social influence (cascade) or external links.
-   Did a particular like spread through social links? 
    -   No, if a user likes a photo and if none of his friends have previously liked the photo.
    -   Yes, if a user likes a photo after at least one of her friends liked the photo  *→* Social cascade.
    -   Example social cascade: A *→* B and A *→ *C *→* E.

Now let's estimate *R**0*from real data. Since *R**0** = d ⋅ q, *need to estimate *q* first: given an infected node count the proportion of its neighbors subsequently infected and average. Then *R**0** = q ⋅ d ⋅ (avg(d**i**2**)/(avg d**i **)**2*where *d**i* is degree of node *i *(last part in the formula is the correction factor due to skewed degree distribution of the network).

Thus, given the start node of a cascade, empirical *R**0*is the count of the fraction of directly infected nodes. 

Authors find that *R**0*correlates across all photos.

![](https://lh4.googleusercontent.com/JFz1jq0PkV9ZCHYs8qg7vFYy9Ud3pAdf1jOFPxPUB-0PN1kFLW50evgiPxt5byKOjMVVRgnJz_M7MEZZ_Eva8lf9z3zWlxKgcmOB3pcYt1KlENhAVoG9CC7_UPEXXHkBa1TNn6Yl)

*Data from top 1,000 photo cascades (each + is one cascade)*

The basic reproduction number of popular photos on Flickr is between 1 and 190. This is much higher than very infectious diseases like measles, indicating that social networks are efficient transmission media and online content can be very infectious. 

##### Epidemic  models

*[Off-top: the lecture was in November 2019 when COVID-19 just started in China -- students were prepared to do analysis of COVID spread in 2020]*

Let virus propagation have 2 parameters:

-   (Virus) Birth rate *β*: probability that an infected neighbor attacks.
-   (Virus) Death rate *δ*: probability that an infected node heals.

General scheme for epidemic models: each node can go through several phases and transition probabilities are governed by the model parameters.

![](https://lh6.googleusercontent.com/67-qZdq2MsTfXvEYKa9YS9TCOLVY3Ay-8suvtHpv7KE-gq_ojR4_V8cop6p2IRWHALcYS-M1jEzR5NNIPYyhDalDP7_mxCPZuCae6AxkG15oGi3rnLBvgZ2-QZNpSEDVPrni-hrH)

*Phases of epidemic models*

###### *SIR model*

-   Node goes through 3 phases: Susceptible -> Infected -> Recovered.
-   Models chickenpox or plague: once you heal, you can never get infected again.
-   Assuming perfect mixing (the network is a complete graph) the model dynamics are:

![ \frac {dS}{dt} = - \beta SI, \frac {dI}{dt} = \beta SI - \delta I, \frac {dR}{dt} = \delta I ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-233b61f7d461d66812a1ca5ccfb9c0c6_l3.svg "Rendered by QuickLaTeX.com")

where *β *is transition probability from susceptible to infected phase, *δ* is transition probability from infected to recovered phase.

![](https://lh5.googleusercontent.com/KKUHb-UJ-Q5QtBv6NW23ao3OHOIb6CO9icFXuTVl7nqJZGh6JILOg7s1T9w65BpIVd-pYzkzG6gKC2_LkXATmPCn7x87TvMnPgapURQ6rj52yQ0iLaVm1qmO-bW5Clm44kN0Q3pt)

*SIR model parameters*

###### *SIS model*

-   Susceptible-Infective-Susceptible (SIS) model.
-   Cured nodes immediately become susceptible.
-   Virus "strength": *s* = *β * / *δ.* 
-   Models flu: susceptible node becomes infected; the node then heals and become susceptible again.
-   Assuming perfect mixing (a complete graph):

![ \frac {dS}{dt} = - \beta SI + \delta I, \frac {dI}{dt} = \beta SI - \delta I ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-e455a27268562acf6f4095e8b1b256b0_l3.svg "Rendered by QuickLaTeX.com")

![](https://lh6.googleusercontent.com/rAne3Pqq4F7bEjQqQiKmVZbiQa_kPy8SutNMh7yy32YO_CZbO7bzWFePGCdbvQHExLku_JPlcL7Q0a4or1NRhrM_X79CBjJEqh_kLOto0mK24IGsTQj6Y2aSfI6p72BideApThXx)

*SIS model parameters*

Epidemic threshold of an arbitrary graph G with SIS model is *τ*, such that: if virus "strength" *s = β / δ < τ* the epidemic can not happen (it eventually dies out).

*τ = 1/ λ1,A* where *λ *is the largest eigenvalue of adjacency matrix A of G.

![](https://lh4.googleusercontent.com/5mtWjuJ03HnBLFEwZmHcdwwHohKOWnq4mRb5kRFViocpSVkxuC69XIhH-wOM0KJqCNsepaY-9MdudCNh337N0EdoRfmleqE_8wQzEACIEKATaK4qFAxN34pIy0veHcPKVMUZwuYE)

*Experiments with SIS model for different parameters*

###### *SEIR model*

Model with phases: Susceptible -> Exposed -> Infected -> Recovered. Ebola outbreak in 2014 is an example of the SEIR model. Paper *Gomes et al., 2014* estimates Ebola's *R0 *to be 1.5-2.

![](https://lh5.googleusercontent.com/uveFAnkwnRKTpvrrnz53ZEmMDxt1jpQjfPNRhaX4y8EYbv2MN1JPkyzaRVChLV0gGR7drg3QIVfzKsOlaWUSA9hbFCtCJxbLKOy_-9gxmn3skWfXN4rCVoxy9SBX1YgW7vmGtncM)

*SEIR model for Ebola case*

###### *SEIZ model*

SEIZ model is an extension of SIS model (Z stands for sceptics).

![](https://lh3.googleusercontent.com/qA8d8iFSXWk-yWyiFVsJEfMN2i30t0wstgnRfj7n1rL8guqU7A_Xw02g4ZngkbKL4dy12N4kj_s1m1xhF3ljdKAgAHsRSSnwGL_5cBW3c1rJ0n2Lo1lFi9IaXbngWCuNHpyDjiNC)

*Transition probabilities of SEIZ model*

Paper *Jin et al. 2013* applies SEIZ for the modeling of News and Rumors on Twitter. They use tweets from eight stories (four rumors and four real) and fit SEIZ model to data. SEIZ model is fit to each cascade to minimize the difference between the estimated number of rumor tweets by the model and number of rumor tweets: *|I(t) -- tweets(t)|.*

To detect rumors, the use new metric (parameters are the same as on figure above):

![ R_{SI} = \frac {(1-p) \beta + (1-l)b}{\rho +\epsilon } ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-71d5816fd5a43e93882b6d9e8d6515b2_l3.svg "Rendered by QuickLaTeX.com")

*RSI* is a kind of flux ratio, the ratio of effects entering E to those leaving E.

All parameters learned by model fitting to real data.

![](https://lh6.googleusercontent.com/o1xx7SMV995VNQy6To6DiJRCaXs-5eg-lcSUuAwzmQPabQR3UUqugFK2uTyHoAnaWF_C3XuC0-tUlXstvP9TohPpf_HDQSt8HaH7TmMmo0SIYnr7F-athAl1hPHzOzvurKyHT1U3)

*Parameters obtained by fitting SEIZ model efficiently identifies rumors vs. news*

###### *Independent Cascade Model*

Initially some nodes *S* are active. Each edge *(u,v)* has probability (weight) *puv*. When node *u* becomes active/infected, it activates each out-neighbor *v* with probability *puv*. Activations spread through the network.

Independent cascade model is simple but requires many parameters. Estimating them from data is very hard *[Goyal et al. 2010]*. The simple solution is to make all edges have the same weight (which brings us back to the SIR model). But it is too simple. We can do something better with exposure curves.

The link from exposure to adoption: 

-   Exposure: node's neighbor exposes the node to the contagion. 
-   Adoption: the node acts on the contagion.

Probability of adopting new behavior depends on the total number of friends who have already adopted. Exposure curves show this dependence.

![](https://lh5.googleusercontent.com/Cc5ZKx2H-QjUdn_5JNE7w6EQhGPpfA26QTlhPrj1JGK_WmWfZud1U5j9z1ZyphTE8GVsGuPAttqqKIMmeaz2d_X2Vae4pkpLymlWbhBcHooC1KDy0IYr8fe4blza4h_dA1g-nrSG)

*Examples of different adoption curves*

Exposure curves are used to show diffusion in viral marketing, e.g. when senders and followers of recommendations receive discounts on products (what is the probability of purchasing given number of recommendations received?) or group memberships spread over the social network (How does probability of joining a group depend on the number of friends already in the group?).

Parameters of the exposure curves:

-   Persistence of *P* is the ratio of the area under the curve *P* and the area of the rectangle of height *max(P)*, width *max(D(P))*.
    -   *D(P) *is the domain of *P*.
    -   Persistence measures the decay of exposure curves.
-   Stickiness of *P* is *max(P)*: the probability of usage at the most effective exposure.

![](https://lh5.googleusercontent.com/V3fE86WyakZzTwSaFtNV4xxzC7mdEdaPV-6J3bI_j9_ycB6YIYRnty70mT0g6GqVUxZTbW16IttH1EJTwMA42IlBqVFj9jdHQTe8z_IvABpSXCYce0X5O1VKnK-uFqf2P3qDw-HM)

*Persistence of P is the ratio of the area under the blue curve P and the area of the red rectangle*

Paper *Romero et al. 2011* studies exposure curve parameters for twitter data. They find that:

-   Idioms and Music have lower persistence than that of a random subset of hashtags of the same size.
-   Politics and Sports have higher persistence than that of a random subset of hashtags of the same size.
-   Technology and Movies have lower stickiness than that of a random subset of hashtags. Music has higher stickiness than that of a random subset of hashtags (of the same size).

* * * * *

Lecture 14 -- Influence Maximization in Networks
===============================================

[*Slides*](http://snap.stanford.edu/class/cs224w-2019/slides/14-influence.pdf)*, *[*Video*](https://youtu.be/oajYN9HKWhI)

Viral marketing is based on the fact that we are more influenced by our friends than strangers. For marketing to be viral, it identifies influential customers, convinces them to adopt the product  (offer discount or free samples) and then these customers endorse the product among their friends.

Influence maximisation is a process (given a directed graph and *k>0*) of finding *k* seeds to maximize the number of influenced people (possibly in many steps).

There exists two classical propagation models: linear threshold model and independent cascade model.

###### *Linear threshold model *

-   A node *v* has a random threshold *θ**v** ~ U[0.1].*
-   A node *v* is influenced by each neighbor *w* according to a weight *b**v,w* such that Σ*b**v,w**≤** 1.*
-   A node *v* becomes active when at least (weighted) *θ**v* fraction of its neighbors are active: Σ*b**v,w**≥** 0.*

###### *Probabilistic Contagion -- Independent cascade model*

-   Directed finite *G = (V, E)**.* 
-   Set *S* starts out with new behavior. Say nodes with this behavior are "active".
-   Each edge *(v,w)* has a probability *p**vw**.*
-   If node *v* is active, it gets one chance to make *w* active, with probability *p**vw**. *Each edge fires at most once. Activations spread through the network.
-   Scheduling doesn't matter. If *u, v* are both active at the same time, it doesn't matter which tries to activate *w* first. But the time moves in discrete steps.

Most influential set of size *k*: set *S* of *k* nodes producing largest expected cascade size *f(S)* if activated. It translates to optimization problem *max* *f(S).* Set *S* is more influential if *f(S)* is larger.

###### *Approximation algorithm* for *influence maximization*

Influence maximisation is NP-complete (the optimisation problem is *max(over S of size k) f(S) *to find the most influential set *S* on *k* nodes producing largest expected cascade size *f(S)* if activated). But there exists an approximation algorithm:

-   For some inputs the algorithm won't find a globally optimal solution/set OPT.
-   But we will also prove that the algorithm will never do too badly either. More precisely, the algorithm will find a set *S* that where *f(S) **≥ 0.63*f(OPT)*, where *OPT* is the globally optimal set.

Consider a Greedy Hill Climbing algorithm to find S.  Input is the influence set *Xu* of each node *u*: *Xu** = {v1**, v2**, ... }.* That is, if we activate *u*, nodes {*v1**, v2*, ... } will eventually get active.The algorithm is as following: at each iteration *i* activate the node *u* that gives largest marginal gain: *max (over u) f(Si-1** ∪ {u}).*

![](https://lh3.googleusercontent.com/leuKBiDfgn0JysKLm64iIEa9wFARewCVIVXd3mDqHnS_zgHuhyk6Nwlg26qGL5QUCbj8ehrCj0y0_eYZ7jk89FprEef49GdJSMHjQfe-y7ACy4kqxjgw94JKUOEw4-yzgdEw9iko)

*Example for (Greedy) Hill Climbing*

For example on the picture above:

-   Evaluate *f({a}) , ... , f({e}),* pick argmax of them.
-   Evaluate *f({**d,**a}) , ... , f({**d,**e})*, pick argmax.
-   Evaluate *f({**d,b,**a}) , ... , f({**d,b,**e})*, pick argmax.

Hill climbing produces a solution *S* where:* f(S) ≥ (1-1/e)*f(OPT) *or* (f(S)≥ 0.63*f(OPT))* (referenced to *Nemhauser, Fisher, Wolsey '78, Kempe, Kleinberg, Tardos '03*). This claim holds for functions *f(-)* with 2 properties (lecture contains proves for both properties, I omit them here): 

1.  *f* is monotone: (activating more nodes doesn't hurt) if *S ⊆ T* then* f(S) ≤ f(T)* and *f({}) = 0*.
2.  *f* is submodular (activating each additional node helps less): adding an element to a set gives less improvement than adding it to one of its subsets:  * ∀ S ⊆  T*

![ f(S \cup \left \{ u \right \}) - f(S) \geq f(T \cup \left \{ u \right \}) - f(T) ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-077f52f8daf1e11c8dc16b9df0d26a12_l3.svg "Rendered by QuickLaTeX.com")

where the left-hand side is a gain of adding a node to a small set, the right-hand side is a gain of adding a node to a large set.

![](https://lh4.googleusercontent.com/jMIx6lpeNjr57c_naBDSA2qbey3HC1V5w7chB-RsEZyvQgy__hGH1crNyC5mCzHh1jGOD73zJnihv2PIrpnbIrECLWDmsvZeESThml768XIE5Za7wTFrdHtovACfhMG_8QXuDeVw)

*Diminishing returns with submodularity*

The bound *(f(S)≥ 0.63 * f(OPT))* is data independent. No matter what is the input data, we know that the Hill-Climbing will never do worse than 0.63**f(OPT).*

Evaluate of influence maximization ƒ(S) is still an open question of how to compute it efficiently. But there are very good estimates by simulation: repeating the diffusion process often enough (polynomial in *n*; *1/ε*). It achieves* (1± ε)-*approximation to *f(S).*

Greedy approach is slow:

-   For a given network *G*, repeat 10,000s of times:
    -   flip the coin for each edge and determine influence sets under coin-flip realization.
    -   each node *u* is associated with 10,000s influence sets *Xui* .
-   Greedy's complexity is *O(k ⋅ n ⋅ R ⋅ m) *where *n* is the number of nodes in *G*,  *k* is the number of nodes to be selected/influenced, *R* is the number of simulation rounds (number possible worlds), *m* is the number of edges in *G*.

###### *Experiment data*

-   A collaboration network: co-authorships in papers of the arXiv high-energy physics theory: 10,748 nodes, 53,000 edges. Example of a cascade process: spread of new scientific terminology/method or new research area.
-   Independent Cascade Model: each user's threshold is uniform random on [0,1]. 
    -   Case 1: uniform probability *p* on each edge.
    -   Case 2: Edge from *v* to *w* has probability *1/deg(w)* of activating *w.*
-   Simulate the process 10,000 times for each targeted set. Every time re-choosing edge outcomes randomly.
-   Compare with other 3 common heuristics:
    -   Degree centrality: pick nodes with highest degree.
    -   Closeness centrality: pick nodes in the "center" of the network.
    -   Random nodes: pick a random set of nodes.
    -   Greedy algorithm outperforms other heuristics (as shown on pictures below).

![](https://lh6.googleusercontent.com/_pnIuXo1hRZ9Feua7VM335oep_HNrNvM8aPEjQ3i7TuCZtEzI3zTmMoaEBZJyRDb3AlsugpCOAQTBHU-IKn1Zke2XMMwMEINslRCptA9neO1RmxkYwH-gC2OxwG82NPpJQxx43ek)

*Uniform edge firing probability puv*

![](https://lh3.googleusercontent.com/wKGOXmtMZzBv2zZYJ76TZXnaDqxQjuvd6i8vGW9P9YKxi2pb_2w8zcRz89lkaG_KJzhNCZhNJmq-oYp8yZmbKr26WkABhcFEH3W8KT2E0xTS2qd-D12JwHm1rTjCQOb7-hltAs_M)

*Non-uniform edge firing probability puv*

##### Speeding things up: sketch-based algorithms

Recap that to perform influence maximization we need to generate a number *R* of possible worlds and then identify *k* nodes with the largest influence in these possible worlds. To solve the problem that for any given node set, evaluating its influence in a possible world takes *O(m) *time (*m* is the number of edges), we will use sketches to reduce estimation time from *O(m)* to *O(1*).

*Idea behind sketches:*

-   Compute small structure per node from which to estimate its influence. Then run influence maximization using these estimates.
-   Take a possible world* G(i)*. Give each node a uniform random number from *[0,1].* Compute the rank of each node *v*, which is the minimum number among the nodes that *v* can reach.
-   Intuition: if *v* can reach a large number of nodes then its rank is likely to be small. Hence, the rank of node *v* can be used to estimate the influence of node *v* a graph in a possible word * G(i)*.

Sketches have a problem: influence estimation based on a single rank/number can be inaccurate:

-   One solution is to keep multiple ranks/numbers, e.g., keep the smallest *c* values among the nodes that *v* can reach. It enables an estimate on union of these reachable sets.
-   Another solution is to keep multiple ranks (say *c* of them): keep the smallest *c* values among the nodes that *v* can reach in all possible worlds considered (but keep the numbers fixed across the worlds).

###### *Sketch-based Greedy algorithm*

Steps of the algorithm:

-   Generate a number of possible worlds.
-   Construct reachability sketches for all node: 
    -   Result: each node has *c* ranks.
-   Run Greedy for influence maximization:
    -   Whenever Greedy asks for the influence of a node set *S*, check ranks and add a *u* node that has the smallest value (lexicographically).
    -   After *u* is chosen, find its influence set of nodes *f(u)*, mark them as infected and remove their "numbers" from the sketches of other nodes.

*Guarantees: *

-   Expected running time is near-linear in the number of possible worlds.
-   When *c* is large, it provides *(1 - 1 / **ε**  - **ε**)* approximation with respect to the possible worlds considered.

*Advantages: *

-   Expected near-linear running time.
-   Provides an approximation guarantee with respect to the possible worlds considered.

*Disadvantage:*

-   Does not provide an approximation guarantee on the "true" expected influence.

![](https://lh4.googleusercontent.com/nWmyQEbVgIxxDQqibkIRliYu6xXAkwqigisgY621zzH5qkxAwxKbfdhmIBCP7WJr4cV3LNs5YID-_v7P3Xkeldw_1DC3HoXVKcu6Sl0ISRKJsamngr0BcQr_coPYFuKCJO-G3qoY)

*Sketch-based achieves the same performance as greedy in a fraction of the time*

* * * * *

Lecture 15 -- Outbreak Detection in Networks
===========================================

[*Slides*](http://snap.stanford.edu/class/cs224w-2019/slides/15-outbreak.pdf)*, *[*Video*](https://youtu.be/V-qd8Thwysk)

The problem this lecture discusses: given a dynamic process spreading over a network we want to select a set of nodes to detect the process effectively. There are *many* *applications*:

-   Epidemics (given a real city water distribution network and data on how contaminants spread in the network detect the contaminant as quickly as possible).
-   Influence propagation (which users/news sites should one follow to detect cascades as effectively as possible?).
-   Network security.

Problem setting for Contamination problem: given graph *G(V, E),* data about how outbreaks spread over the for each outbreak *i* we know the time *T(u,i)* when outbreak *i* contaminates node *u. *The goal is to select a subset of nodes *S *that maximizes the expected reward for detecting outbreak *i:* *max f(S) = Σ p(i) fi(S) *subject to *cost(S) < B *where *p(i) *is the probability of outbreak* i *occurring*, f(i) *is the reward for detecting outbreak* i *using sensors* S.*

Generally, the problem has two parts:

1.  Reward (one of the following three): 
    1.  Minimize time to detection.
    2.  Maximize number of detected propagations.
    3.  Minimize the number of infected people.
2.  Cost (context dependent): 
    1.  Reading big blogs is more time consuming.
    2.  Placing a sensor in a remote location is expensive.

![](https://lh6.googleusercontent.com/JzZQDpeZ2gkaFc4l2wLq7toeVZrGkgfNhagP6NkKeOm6nh73xb42sYELw89kp-kIvcKn3aei8ndfLWr0xWmMjKjMNshvvyb4VRhVUqo3d901shwPuAHnI-pUL865rUVgVlaEXj9c)

*Monitoring blue node saves more people than monitoring the green node*

Let set a penalty *π**i**(t)* for detecting outbreak *i* at time *t*. Then for all three reward settings  detecting sooner does not hurt:

1.  Time to detection (DT): how long does it take to detect a contamination? 
    -   Penalty for detecting at time *t*: *πi**(t)* *= t.*
2.  Detection likelihood (DL): how many contaminations do we detect? 
    -   Penalty for detecting at time *t*: *πi**(t)* *= 0*, *πi**(∞) = 1.* Note, this is a binary outcome: we either detect or not.
3.  Population affected (PA): how many people drank contaminated water? 
    -   Penalty for detecting at time *t*: *πi**(t)* *=* {# of infected nodes in outbreak *i* by time *t*}.

Now, let's define *f**i**(S)* as penalty reduction: *f**i**(S) = **π**i**(∞) - **π**i**(T(S, i))*. With this we can observe diminishing returns:

![](https://lh4.googleusercontent.com/zW-PpT_84OtoOatJ9Ru-cDsYKlmB94M9YSJDhCUeIf2YPWONyEarfJmgnl7iyKNzE_ywhFJgZqJO4fSZqzFWbIf9QixiHN8OLSovE1ZUWR9DGB2bEeLTDe-Vrx9BxZuD-byXJtU5)

*Diminishing returns example*

Now we can see that objective function is submodular (recall from previous lecture:  *f* is submodular if activating each additional node helps less).

What do we know about optimizing submodular functions? 

-   Hill-climbing (i.e., greedy) is near optimal: *(1 - 1 / e ) ⋅ OPT*.
-   But this only works for unit cost case (each sensor costs the same):
    -   For us each sensor *s* has cost *c(s)*.
-   Hill-climbing algorithm is slow: at each iteration we need to re-evaluate marginal gains of all nodes.
    -   Runtime *O(|V| - K)* for placing *K* sensors.

###### *Towards a new algorithm*

Consider the following algorithm to solve the outbreak detection problem: Hill-climbing that ignores cost:

-   Ignore sensor cost *c(s).*
-   Repeatedly select sensor with highest marginal gain.
-   Do this until the budget is exhausted.

But it can fail arbitrarily badly. There exists a problem setting where the hill-climbing solution is arbitrarily far from *OPT*. Next we come up with an example.

*Bad example when we ignore cost: *

-   *n* sensors, budget *B.*
-   *s1*: reward *r*, cost *c*.
-   *s2...sn*: reward* r - ε, c = 1.* 
-   Hill-climbing always prefers more expensive sensor *s1* with reward *r* (and exhausts the budget). It never selects cheaper sensors with reward *r - ε* → For variable cost it can fail arbitrarily badly.

Bad example when we optimize benefit-cost ratio (greedily pick sensor *si* that maximizes benefit to cost ratio):

-   Budget *B.*
-   2 sensors *s**1* and *s**2*: costs *c(s**1**)* = *ε*, *c(s**1**)* *= B*; benefit (only 1 cascade): *f(s**1**) = 2**ε*, *f(s**2**)* *= B.*
-   Then the benefit-cost ratio is:* f(s**1**) / c(s**1**) = 2* and *f(s**2**) / c(s**2**) = 1*. So, we first select *s**1* and then can not afford *s**2*  → We get reward *2**ε* instead of *B*. Now send *ε* *→ 0* and we get an arbitrarily bad solution.
-   This algorithm incentivizes choosing nodes with very low cost, even when slightly more expensive ones can lead to much better global results.

The solution is the CELF (Cost-Effective Lazy Forward-selection) algorithm. It has two passes: set (solution) *S′* -- use benefit-cost greedy and Set (solution) *S′′* -- use unit-cost greedy. Final solution: *S = arg max ( f(S'), f(S"))*. 

CELF is near optimal *[Krause&Guestrin, '05]: *it achieves *½(1-1/e)* factor approximation. This is surprising: we have two clearly suboptimal solutions, but taking best of the two is guaranteed to give a near-optimal solution.

###### *Speeding-up Hill-Climbing: Lazy Evaluations*

Idea: use *δ**i* as upper-bound on *δ**j* *(j > i)*. Then for lazy hill-climbing keep an ordered list of marginal benefits *δ**i* from the previous iteration and re-evaluate *δ**i* only for top node, then re-order and prune.

CELF (using Lazy evaluation) runs 700 times faster than greedy hill-climbing algorithm:

![](https://lh5.googleusercontent.com/bLSzHJCmKnGzNI9-uumC5yvZ-gXcnUHYM1TumAPsH6VkLVE4YbtbH1pc95RffMR34HGi6J9MzDsZ1KHgk6GEZ122g3weIJNH4cAjeGDhwTEwtJHRkPKXqAf2T3LxaZBu-peHhFQf)

*Scalability of SELF (CELF+bounds is CELF together with computing the data-dependent solution quality bound)*

###### *Data Dependent Bound on the Solution Quality*

The* (1-1/e)* bound for submodular functions is the worst case bound (worst over all possible inputs). Data dependent bound is a value of the bound dependent on the input data. On "easy" data, hill climbing may do better than 63%. Can we say something about the solution quality when we know the input data?

Suppose *S* is some solution to *f(S)* s.t. *|S| ≤ k* ( *f(S)* is monotone & submodular):

-   Let *OPT = {t**i**, ... , t**k**}* be the OPT solution.
-   For each *u* let *δ(u)** = f(S ∪ {u} - f(S).*
-   Order *δ(u)* so that *δ(1)* ≥ *δ(2)* ≥ ⋯ 
-   Then:* f(OPT) ≤ f(S) + ∑**δ(i)*. 
-   Note: this is a data dependent bound (*δ(i)* depends on input data). Bound holds for any algorithm. It makes no assumption about how *S* was computed. For some inputs it can be very "loose" (worse than 63%).

###### *Case Study: Water Network*

Real metropolitan area water network with *V* = 21,000 nodes and *E* = 25,000 pipes. Authors *[Ostfeld et al., J. of Water Resource Planning]* used a cluster of 50 machines for a month to simulate 3.6 million epidemic scenarios (random locations, random days, random time of the day).

Main results:

-   Data-dependent bound is much tighter (gives more accurate estimate of algorithmic performance).
-   Placement heuristics perform much worse.
-   CELF is much faster than greedy hill-climbing (but there might be datasets/inputs where the CELF will have the same running time as greedy hill-climbing):

![](https://lh5.googleusercontent.com/lEYsLtoFvQEDZr9TSPKpgeumg-ySh2uC1XK2i5fATCKKHfhHLmAZk5kC4TuxBDeBm-D2xbeuntMaSEDT5SrzOXVZlUBIn-oAvAQij5L9un3qAyO2ES26zXHNCOSkP8E94U0qwlUA)

*Solution quality*

-   Different objective functions give different sensor placements:

![](https://lh3.googleusercontent.com/kFHKFqLkSnd3pF-xu86WMGVWazx03Tb3BF7QCYT2MZooNWc9MEwyP6eVGfYb2q7tZPZnLsoo_IDlDsrhm-cbezbMKCauz27Zqu0vR_Uf2BCMTyZC6cjmH0LXu8vTvlj7thmXqDXX)

*Placement visualization for different objectives*

###### *Case Study: Cascades in blogs*

*Setup:*

-   Crawled 45,000 blogs for 1 year.
-   Obtained 10 million news posts and identified 350,000 cascades.
-   Cost of a blog is the number of posts it has.

*Main results:*

-   Online bound turns out to be much tighter: 87% instead of 32.5%.
-   Heuristics perform much worse: one really needs to perform the optimization.
-   CELF runs 700 times faster than a simple hill-climbing algorithm.

![](https://lh3.googleusercontent.com/nMsTxkjANooORzx64k2l8BIMxhUV50qevXoo4Phz0JRJjlNc-qyxwz8PcUl-b8PhHi4uZgss7JOi7mhBkm_yCHfBASKvGOzimFnX4QZT8gV0UmuZeBAwIMVkG-guhtPMUIunFLxK)

Solution quality

CELF has 2 sub-algorithms:

-   Unit cost: CELF picks large popular blogs.
-   Cost-benefit: cost proportional to the number of posts.
-   We can do much better when considering costs.
-   But there is a problem: CELF picks lots of small blogs that participate in few cascades. Thus, we pick best solution that interpolates between the costs -> we can get good solutions with few blogs and few posts.

We want to generalize well to future (unknown) cascades. Limiting selection to bigger blogs improves generalization:

![](https://lh6.googleusercontent.com/Wyr8rPuGaqZ-B_-8NFrULgQ3prJ5kEZD6GD6KPuFIZfkIutWH6-x7KSWN3vXIzk4pp46mLBCbzPm_K9jYWWmLnRECocDOZMvKODDl1wPxeYC_KDQGmPhmT-e9M3Li_a2ymBMcDiL)

*Generalization to Future*

* * * * *

Lecture 16 -- Network Evolution
==============================

[*Slides*](http://snap.stanford.edu/class/cs224w-2019/slides/16-evolution.pdf)*, *[*Video*](https://youtu.be/-7hKWLhHk1w)

Evolving networks are networks that change as a function of time. Almost all real world networks evolve either by adding or removing nodes or links over time. Examples are social networks (people make and lose friends and join or leave the network), internet, web graphs, e-mail, phone calls, P2P networks, etc.

The picture below shows the largest components in Apple's inventor network over a 6-year period. Each node reflects an inventor, each tie reflects a patent collaboration. Node colors reflect technology classes, while node sizes show the overall connectedness of an inventor by measuring their total number of ties/collaborations (the node's so-called degree centrality).

![](https://lh4.googleusercontent.com/k_qV5yZdG3kUlyDEvBaNGFianSnWZFJ1qelQMPuZTF9_pxCrw334aGtiNy-1Tm6krhZc1pZP-glYn3RsFhOWbbJjzA89EvDqtM_qbTmuEugFqm0UeSyXDLJX7K-xP4RGhwTuPiU0)

*The largest components in Apple's inventor network *

There are three levels of studying evolving networks:

1.  Macro level (evolving network models, densification).
2.  Meso level (Network motifs, communities).
3.  Micro level (Node, link properties -- degree, network centrality).

###### *Macroscopic evolution of networks *

The questions we answer in this part are:

1.  What is the relation between the number of nodes *n(t)* and number of edges *e(t)* over time *t*?  
2.  How does diameter change as the network grows? 
3.  How does degree distribution evolve as the network grows?

Q1: Let's set at time *t* nodes *N(t)*, edges *E(t)* and suppose that *N(t+1)* = 2 ⋅ *N(t)*. What is now *E(t+1)*? It is more than doubled. Networks become denser over time obeying Densification Power Law: *E(t) ∝ N(t)a *where* a *is densification exponent (*1 ≤ a ≤ 2*). In other words, it shows that the number of edges grows faster than the number of nodes -- average degree is increasing. When *a=1*, the growth is linear with constant out-degree (traditionally assumed), when *a=2*, the growth is quadratic -- the graph is fully connected.

Q2: As the network grows the distances between the nodes slowly decrease, thus diameter shrinks over time (recap how we compute diameter in practice: with long paths, take 90th-percentile or average path length (not the maximum); with disconnected components, take only the largest component or average only over connected pairs of nodes).

But consider densifying random graph: it has increasing diameter (on picture below). There is more to shrinking diameter than just densification.

![](https://lh6.googleusercontent.com/mXI4lpN9ZsxYOWJ95giKiN4c_7b9ZhrOg00iKxZ6M56ACm2t7nE33R7o9EbQys8f8vx9qsPG82Lku-6Y4_EMl0KcPU0vy5oIRr7e1IWdKD-p73qNFFsM7GgGvJ3ziHQANEWfvDkE)

*Diameter of Densifying Gnp*

Comparing rewired random network to real network (all with the same degree distribution) shows that densification and degree sequence gives shrinking diameter.

![](https://lh4.googleusercontent.com/ZjmwznadITPjKxYtZs_3CR5Xb7kJVC1iMJESwoAdX_JAuZngGcvWHzsyUXfSDbOHuLjtfSpeoTQMKbL5y1hQVyFtsXelHcLcvKhi9FM5TTr1WwnVl6AGTVFVpemH4pUQ9ujlcN0G)

*Real network (red) and random network with the same degree distribution (blue)*

But how to model graphs that densify and have shrinking diameters? (Intuition behind this: How do we meet friends at a party? How do we identify references when writing papers?)

###### *Forest Fire Model*

The Forest Fire model has 2 parameters: *p* is forward burning probability,  *r* is backward burning probability. The model is Directed Graph. Each turn a new node *v* arrives. And then:

1.  *v* chooses an ambassador node *w* uniformly at random, and forms a link to *w.*
2.  Flip 2 coins sampled from a geometric distribution: generate two random numbers *x* and *y* from geometric distributions with means *p / (1 - p)* and *rp / (1 - rp).*
3.  *v* selects *x* out-links and *y* in-links of *w* incident to nodes that were not yet visited and form out-links to them (to "spread the fire" along).
4.  *v* applies step (2) to the nodes found in step (3) ("Fire" spreads recursively until it dies; new node *v* links to all burned nodes).

![](https://lh5.googleusercontent.com/tbeB0LtjobNBZAXjOB0k6ALDGh88m-rwxK2M1WAXO0zW5cFjk1gGx1Q4xoF5a4tMl1uqjiTW3Pq4Ug6vVGrNks6DHfr3-aj6UZPgt5RiOizJjoltxPmqWtLKC_2q9Tet-oHsD1wx)

*Example of Fire model*

On the picture above:

1.  Connect to a random node *w*.
2.  Sample *x = 2, y = 1.*
3.  Connect to 2 out- and 1 in-links of *w*, namely *a,b,c.*
4.  Repeat the process for *a,b,c.*

In the described way, Forest Fire generates graphs that densify and have shrinking diameter:

![](https://lh4.googleusercontent.com/tiX0E8PCGJS7woK6ITk1DCak0IPTrYQJDwNLUASpCBB3bRGkuO6b-V8rl4sDmERkYWT9aNdoPgR0FSi3t-WnqCOHT4ZCJyrOsDXoTOpEXPyuxuUWF4dyR6UT8R7C7pSTiRvFaWax)

Also, Forest Fire generates graphs with power-law degree distribution:

![](https://lh3.googleusercontent.com/NKL1aPnxIBDyVAQXacxWuyGudSr9ZkxegkViwWFIjv8ESBH0jpdf8M4vWtM0eEZM3fzbVFxrcpGpM63Uw3prybbg7FT2vExjG6JCfVDAKcig1yHvy2xPdrzvQCtRDq9Gl01g8wRI)

We can fix backward probability *r* and vary forward burning probability *p.* Notice a sharp transition between sparse and clique-like graphs on the plot below. The "sweet spot" is very narrow.

![](https://lh3.googleusercontent.com/LwaP3GuqXtWuoAcvhG3IMEDB9XLCM10qJJtsVYU8DZaS2yqFqWRrKWfLukeuGUl2C0XrwYXh4BGqJtx6-NSSzGh4LkXEUOS7BJudsxLe5q4YZ0ynRRCoj6O2TawRSRwA7yxnLV_b)

*Phase transition of Forest Fire model*

###### *Temporal Networks*

Temporal network is a sequence of static directed graphs over the same (static) set of nodes *V.* Each temporal edge is a timestamped ordered pair of nodes *(ei** = (u, v) , ti**)*, where *u, v ∈ V* and *ti* is the timestamp at which the edge exists, meaning that edges of a temporal network are active only at certain points in time. 

![](https://lh3.googleusercontent.com/8iDoD8YaHy6Afi6d_i3OY1LAF6rVizVUaX2eVLG-f01mycUAlP2CqGet79OZF05e8XDBnSl8n7YpEWlqnrvOlORcpgop115yEkfy9ARGvUUe5BnNSZe27eczZT5Ubf419eOLoZO4)

*Temporal network*

*Temporal network examples:*

-   Communication: Email, phone call, face-to-face.
-   Proximity networks: Same hospital room, meet at conference, animals hanging out.
-   Transportation: train, flights.
-   Cell biology: protein-protein, gene regulation.

###### *Microscopic evolution of networks *

The questions we answer in this part are:

-   How do we define paths and walks in temporal networks? 
-   How can we extend network centrality measures to temporal networks?

###### Q1:* Path*

A temporal path is a sequence of edges *(u**1**, u**2**, t**1**), (u**2**, u**3**, t**2**), ... , (u**j**, u**j+1**, t**j**)* for which *t**1* ≤ *t**2* ≤ ⋯ ≤ *t**j* and each node is visited at most once.

![](https://lh3.googleusercontent.com/-kGRyYfaagUzoWujFUJxeTmlISxSpBexsSLiM6b4rhUnJovf1bXIpL0wteyy3BJiAVYy8ljcBQMimExakduI44NwvVLk5Kjp4wr_unZFgahuUg4EPp66JzFivN5GBlAazJuPcFwQ)

*The sequence of edges [(5,2),(2,1)] together with the sequence of times t1, t3 is a temporal path*

To find the temporal shortest path, we use the TPSP-Dijkstra algorithm -- an adaptation of Dijkstra using a priority queue. Briefly, it includes the following steps:

-   Set distance to ∞ for all nodes.
-   Set distance to *0* for *n**s* (source node).
-   Insert (nodes, distances) to *PQ* (priority queue).
-   Extract the closest node from *PQ*.
-   Verify if edge *e* is valid at *t**q*(time of the query -- we calculate the distance from source node *n**s* to target node *n**t* between time *t**s* and time *t**q*)*.*
-   If so, update *v*'s distance from *n**s*.
-   insert* (v, d[v])* to *PQ* or update *d[v]* in *PQ *where *d[v] *is the distance of *n**s** to *v.

![](https://lh5.googleusercontent.com/6GIztZTqipOe6uAXP6W2mAhzlq0SRFvS7jl-zzPJn8e-uYjPnP2Z1Qv8EmYX1nAFFHsyT-dyyBwWK8OZulRPwwpuiU_EgS980XtXArNHZdcUOjj13BUXGEGz1IwlbCsumH1cUcaT)

*Example of a temporally evolving graph. Shortest paths from a to f are marked in thick lines.*

###### Q2: *Centrality*

Temporal closeness is the measure of how close a node is to any other node in the network at time interval *[0,t]*. Sum of shortest (fastest) temporal path lengths to all other nodes is:

![ c_{clos}(x,t) = \frac{1}{\sum_y{d(y,x|t)}} ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-8b22825252504fb3f0f2b0d14c12d030_l3.svg "Rendered by QuickLaTeX.com")

where denominator is the length of the temporal shortest path from *y* to *x* from time 0 to time *t.*

Temporal PageRank idea is to make a random walk only on temporal or time-respecting paths.

-   A temporal or time-respecting walk is a sequence of edges *(u**1**, u**2**, t**1**), (u**2**, u**3**, t**2**), ... , (u**j**, u**j+1**, t**j**)* for which *t**1* ≤ *t**2* ≤ ⋯ ≤ *t**j**.* 

As *t* → ∞, the temporal PageRank converges to the static PageRank. Explanation is this:

-   Temporal PageRank is running regular PageRank on a time augmented graph: 
    -   Connect graphs at different time steps via time hops, and run PageRank on this time-extended graph.
    -   Node *u* at *ti* becomes a node *(u,  ti**)* in this new graph.
    -   Transition probabilities given by *P ((u, ti** ), (x, t2** )) | (v, t0**, (u, ti** ) = β| Гu |* .
-   As *t* → ∞, *β| Гu | * becomes the uniform distribution: graph looks as if we superimposed the original graphs from each time step and we are back to regular PageRank.

How to compute temporal PageRank:

-   Initiating a new walk with probability *1 - α*.
-   With probability *α* we continue active walks that wait in *u.*
-   Increment the active walks (active mass) count in the node *v* with appropriate normalization *1-β.*
    -   Decrements the active mass count in node *u.*
-   Increments the active walks (active mass) count in the node *v* with appropriate normalization *1-β*.
    -   Decrements the active mass count in node *u.*

In math words, temporal PageRank is:

![ r(u,t) = \sum_{v \in C}^{} \sum_{k=0}^{t}(1-\alpha) \alpha ^{k} \sum_{z \in Z(v,u|t), |z|=k}P[z|t] ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-9fd1f85c3c93b12f62169f147ddc6954_l3.svg "Rendered by QuickLaTeX.com")

where *Z(v,u | t )* is a set of all possible temporal walks from *v* to *u* until time *t* and *α* is the probability of starting a new walk.

Case Studies for temporal PageRank:

-   Facebook: A 3-month subset of Facebook activity in a New Orleans regional community. The dataset contains an anonymized list of wall posts (interactions).
-   Twitter: Users' activity in Helsinki during 08.2010-- 10.2010. As interactions we consider tweets that contain mentions of other users.
-   Students: An activity log of a student online community at the University of California, Irvine. Nodes represent students and edges represent messages.

*Experimental setup: *

-   For each network, a static subgraph of n = 100 nodes is obtained by BFS from a random node.
-   Edge weights are equal to the frequency of corresponding interactions and are normalized to sum to 1.
-   Then a sequence of 100K temporal edges are sampled, such that each edge is sampled with probability proportional to its weight.
-   In this setting, temporal PageRank is expected to converge to the static PageRank of a corresponding graph.
-   Probability of starting a new walk is set to *α* = 0.85, and transition probability *β* for temporal PageRank is set to 0 unless specified otherwise.

Results show that rank correlation between static and temporal PageRank is high for top-ranked nodes and decreases towards the tail of ranking:

![](https://lh3.googleusercontent.com/IkBjK705lojLmykrpt_atcY5Vm8-HqU8g-_SBWTWfktaVHMx12SMkiwPz0xDvyVXBLxeyf15_7d4GgDd9_Po3qd_oPQdM7-2FzHY45aOhZmceNWqtYekc3X2Sevg4T4y31xwHEZO)

*Comparison of temporal PageRank ranking with static PageRank ranking*

Another finding is that smaller *β* corresponds to slower convergence rate, but better correlated rankings: 

![](https://lh6.googleusercontent.com/6HHT7dY4QVd9ZlxLtU_cPhyBusUHxrIuUSPB8T8IyfYo9ZXPcgdGhTIWhT8ivQZD0m8V3GsoNK-nAI-GNe6cPcZWg9akSzN97a6XSA7TAV86_1suDSXQ180uAJ4xc4kMslMKjRju)

*Rank quality (Pearson correlation coefficient between static and temporal PageRank) and transition probability β*

###### *Mesoscopic evolution of networks *

The questions we answer in this part are:

-   How do patterns of interaction change over time? 
-   What can we infer about the network from the changes in temporal patterns?

###### *Q1: Temporal motifs*

*k*-node *l*-edge *δ*-temporal motif is a sequence of *l* edges *(u**1**,v**1**, t**1**), (u**2**,v**2**, t**2**), ..., (u**l**,v**l**, t**l**)* such that *t**1* < *t**2* < ... < *t**l* and *t**l* -- *t**1* ≤ *δ*. The induced static graph from the edges is connected and has *k* nodes.

Temporal motifs offer valuable information about the networks' evolution: for example, to discover trends and anomalies in temporal networks.

*δ-*temporal Motif Instance is a collection of edges in a temporal graph if it matches the same edge pattern, and  all of the edges occur in the right order specified by the motif, within a *δ* time window.

![](https://lh5.googleusercontent.com/2AR5FQcXQ9EcfdtekeBX3cdOA7s_L1tLCVnQ-Jy6maMEVkljrQRCLUAYuo6SIaO_fUefkfskd3I9iHg4pjsQ_6gLBI7C-qWi0ZWD9df3jWc3t82OoOlbpSbkkLrABkOEdgLPxgE9)

*Example of temporal motif instances*

###### Q2: *Case Study -- Identifying trends and anomalies*

Consider all 2- and 3- node motifs with 3 edges:

![](https://lh6.googleusercontent.com/fwZjHQz_RNdG2ii1yYNGzm21vqunuxRhUvz71fiPwyYmPkRm1rB7e_q5an5VurILO19nzVKCJH2Pi-zcPRqprzymiG-Ota3GiTzDXsWdIv9nnPqe1ZqDyHmeIylz2bdxFyVrx7q2)

*The green background highlights the four 2-node motifs (bottom left) and the grey background highlights the eight triangles.* 

The study *[Paranjape et al. 2017] *looks at 10 real-world temporal datasets. Main results are:

-   Blocking communication (if an individual typically waits for a reply from one individual before proceeding to communicate with another individual): motifs on the left on the picture below capture "blocking" behavior, common in SMS messaging and Facebook wall posting, and motifs on the right exhibit "non-blocking" behavior, common in email.

![](https://lh3.googleusercontent.com/1F8QGc2T2r9yRWxmjCxGBfzQ_LAsKm0sH_gVsP5dh2osLd-4xPYOqulcl6OTovOj8j58sHTXJuAwdYcVjE5U6U5bF3-cYRz1yrc6OBGVDFDvmHaxUPBekEQ7UaWwgi3TIRtyzmy5)

*Fraction of all 2 and 3-node, 3-edge δ-temporal motif counts that correspond to two groups of motifs (δ = 1 hour).*

-   Cost of Switching:
    -   On Stack Overflow and Wikipedia talk pages, there is a high cost to switch targets because of peer engagement and depth of discussion.
    -   In the COLLEGEMSG dataset there is a lesser cost to switch because it lacks depth of discussion within the time frame of δ = 1 hour.
    -   In EMAIL-EU, there is almost no peer engagement and cost of switching is negligible

![](https://lh5.googleusercontent.com/FY3w-IEzM-nh53lzQj9Ska2pJPLnkUyt32Oze8Qqp3taWZO6EdUsd9fM_INjnz3kSyUjHHLVCaiJEDau6EzhlZnYUMyVNv-hl21Zsb0U4XldI24BHRVpbxRnAsUQNITlX3590Oub)

*Distribution of switching behavior amongst the nonblocking motifs (δ = 1 hour)*

###### *Case Study -- Financial Network*

To spot trends and anomalies, we have to spot statistically significant temporal motifs. To do so, we must compute the expected number of occurrences of each motif:

-   Data: European country's transaction log for all transactions larger than 50K Euros over 10 years from 2008 to 2018, with 118,739 nodes and 2,982,049 temporal edges (*δ*=90 days).
-   Anomalies: we can localize the time the financial crisis hits the country around September 2011 from the difference in the actual vs. expected motif frequencies.

![](https://lh3.googleusercontent.com/eiZNiUEYxqI0DN4lDHqQBHbRyKCQvqZiGiB1G5yRMcfqSdR8TALtWqnw1FeauBJeZjKXcEMgKUEVJk38jOLqwfOXzDDINehO20nSmBDkQIs_Ipr2xX_OpOPRgctFlfn7jTzDzCmJ)

*Differences between actual and expected motifs (red lines indicate when financial crisis started)*

