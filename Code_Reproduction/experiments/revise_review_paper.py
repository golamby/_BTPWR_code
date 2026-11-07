"""Update manuscript from the checked campaign tables and compress obsolete appendix results."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'Paper_Source'
tables=ROOT/'final_results'/'tables'
df=pd.read_csv(tables/'primary_cases.csv')
order=['BTPWR','BTPWR-NP','C-seeded','Mapped','ETBP-P','Greedy-C/R','Random','BPQA numeric']
means=df.groupby('method').mean(numeric_only=True)
rows=[]
for name in order:
    x=means.loc[name]
    rows.append(f"{name} & {x.completion:.2f} & {x.answered:.2f} & {x.ce3:.3f} & {x.payment/1000:.2f} & {x.physical/1000:.2f} \\\\")
table='\n'.join(rows)
main=r'''
\section{Experimental Evaluation}\label{sec:eval}\label{sec:recovery-evaluation}
\subsection{Setup and Comparisons}
T-Drive, Foursquare-TKY and Brightkite supply mobility-derived observation
masks~\cite{yuan10,yang16,cho11}. These are transferred to the supplied
controlled humidity field: 30 columns are public history and the following
50 are collection rounds. The field lacks verifiable provider and time
metadata, so this is a controlled numerical replay. A campaign contains 100
workers and 100 regions, with 50 requested regions per round. Workers make
offers on their actual routes; their latent behavior and target field are not
revealed to recruitment. Honest and malicious intrinsic reliabilities are
sampled once from $[.8,1]$ and $[0,.3)$, respectively. Honest reports have
checking-scale Gaussian noise with standard deviation at most $.1$; malicious
reports are uniform on $[10,40]$.

Initially $70\%$ of workers are honest and $10\%$ of the whole pool are
accurately certified honest, at no acquisition cost. Their estimated scores
are sampled uniformly from $[.95,1]$; all other scores start at $.5$.
Defaults are $B=450$ per round, $(\epsilon_1,\epsilon_2)=(2.1,2.7)$,
$\gamma=.6$ and $(\theta_C,\theta_M)=(.8,.3)$. The common 50-round cap is
an upper bound on each round's commitment; unspent funds do not augment the
next round's cap. All selected routes incur physical sampling cost, including
reports that fail a method's acceptance test. Commitment and realized
settlement are recorded separately.

The primary comparison uses eight new paired seeds (9801--9808) on each
mobility pattern, for 24 campaigns per method. Methods share potential
reports, tasks, offers, intrinsic types and caps, but evolve their own state.
The 95\% paired Student intervals first average three mobility results within
each seed; they describe these fixed field and trace conditions. Parameter
sweeps use seeds 9701--9703, all three mobility patterns, and vary one factor
at a time: worker counts 50--100 by 10, per-round cap 250--600 by 50,
honest fraction 50--90\% by 10 points, certified fraction 10--30\% by 10
points, and $\epsilon_1\in\{1.3,1.5,1.7,1.9,2.1\}$. All other settings stay
at their defaults.

The BTPWR-NP, C-seeded and Mapped controls share the input, feedback and
recovery pipeline but change recruitment. ETBP-P adapts the priority rule of
ETBP-TD~\cite{etbp25} to that pipeline and cap; it is not the original full
system. Random orders offers randomly and Greedy-C/R orders by marginal
released coverage times estimated reputation per charge, both under the
same pipeline. BPQA numeric implements the recruitment, critical payment,
range check, truth discovery and reputation mechanisms described by
BPQA~\cite{bpqa26} and its cited auction. It receives physical incidence and
bids and uses its own truth discovery, without BTPWR recovery; the result
therefore compares numerical systems rather than the selector in isolation.
Sparse-report and normalization choices, and the absence of a measured
cryptographic deployment, are specified in the appendix.

\subsection{Metrics and Primary Results}
A task is successful when it is answered and its absolute error is below
$10\%$ of the magnitude of truth (exact agreement for zero truth).
Completion divides successful tasks by all requested tasks; answer rate
counts finite answers. MAE and RMSE compare each method on the exact
intersection of tasks answered by it and BTPWR. In addition,
\begin{equation}\label{eq:evaluation-metrics}
\mathrm{CE}(P)=\frac{\sum_t[\sum_{h\in\mathcal S_t} |e_{ht}|+
P(|\mathcal T_t|-|\mathcal S_t|)]}{\sum_t|\mathcal T_t|},
\end{equation}
where $\mathcal S_t$ is the successful set. We use CE(3) as an auxiliary
score: its penalty is a chosen exchange rate, so completion, answer rate and
error remain separate. A worker is evaluated as identified honest at score
$\ge .8$ and identified malicious below $.3$; these common evaluation
thresholds need not coincide with another method's own decision boundary.

\begin{table*}[t]\centering\footnotesize
\caption{Primary numerical comparison: means over 24 paired campaigns. Payment and physical cost are totals over 50 rounds, in thousands of units.}
\label{tab:primary-review}
\begin{tabular}{lrrrrr}\toprule
Method & Completion (\%) & Answer rate (\%) & CE(3) & Commitment & Physical cost\\\midrule
@TABLE@
\bottomrule\end{tabular}
\end{table*}

Table~\ref{tab:primary-review} gives the unconditional campaign means.
Against C-seeded, BTPWR uses $6.55\%$ less committed payment (95\% CI
$[-7.43,-5.66]\%$) and $12.71\%$ less physical cost
($[-14.83,-10.59]\%$), while completion is $1.63$ percentage points lower
($[-3.01,-0.24]$). Against BTPWR-NP, its completion is $1.05$ points higher
($[-0.06,2.16]$), with $5.22\%$ more commitment and $9.12\%$ more physical
cost. Against ETBP-P the completion difference is $0.22$ points
($[-0.90,1.34]$), with $3.38\%$ lower commitment and $5.03\%$ higher physical
cost. These are paired contrasts, not equivalence conclusions. BTPWR exceeds
Random by $16.75$ completion points under the common pipeline.

The numerical BPQA system completes $27.28\%$ of requests and BTPWR
$76.17\%$ on these worlds; the paired difference is $48.89$ points
($[47.43,50.35]$). BPQA commits less payment on average, while BTPWR has
$40.60\%$ lower physical cost in the paired comparison. The two systems
use different information and recovery services; these results do not
attribute the gap to any single component or compare cryptographic overhead.

\begin{figure*}[t]\centering
\includegraphics[width=.96\textwidth]{figures/quality_sensitivity.pdf}
\caption{Sensitivity of completion, CE(3), and paired common-task RMSE to
budget, population and honest fraction. Each point averages three mobility
patterns and three paired seeds. Shading gives the 95\% seed-level interval
for BTPWR; RMSE differences are comparator minus BTPWR on their shared tasks.}
\label{fig:review-quality}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=.96\textwidth]{figures/recognition_sensitivity.pdf}
\caption{Completion, honest identification and malicious acceptance under
certification, honest-population and location-privacy settings. BPQA uses a
different native privacy input and is omitted from the privacy-parameter
axis. Recognition thresholds are common evaluation thresholds, rather than
claims that every method has the same internal classification rule.}
\label{fig:review-recognition}
\end{figure*}

\subsection{Sensitivity, Recognition and Evidence Isolation}
Figs.~\ref{fig:review-quality} and~\ref{fig:review-recognition} show the
parameter grid rather than a single operating point. Budget raises service
coverage, while the behavior composition affects the systems differently.
In the primary campaigns BTPWR identifies $35.1\%$ of honest workers and
$21.2\%$ of malicious workers by the final round; C-seeded identifies
$63.1\%$ and $60.4\%$, respectively. The latter uses more payment and
physical sampling. Initially certified workers count as identified honest;
unobserved workers cannot be assumed identified. The common-threshold
malicious acceptance rate is zero for BTPWR and $46.4\%$ for BPQA numeric,
whose score dynamics and operating threshold differ. These are classification
outcomes under the stated behavior model, not a general adversary guarantee.
Resource and round-by-round curves are in the supplement.

A separate $2\times2$ synthetic experiment crosses BTPWR's planner and a
coverage--reliability selector with separated versus recovered-value-coupled
feedback. It covers 32 worlds across two patterns, eight seeds and two
historical caps. With BTPWR selection, coupling lowers completion by
$11.88$ points (95\% CI $[-14.80,-8.97]$); under the alternative selector
it lowers completion by $13.18$ points ($[-15.10,-11.27]$). The interaction
interval includes zero. These deliberately coupled arms are synthetic
controls rather than BPQA's full system, and the result describes this
intervention under the stated conditions.

'''.replace('@TABLE@',table)
path=PAPER/'main.tex';text=path.read_text(encoding='utf-8')
text=text.replace(text[text.index('Given accurate initial certification, a $2\\times2$ factorial'):text.index('\\end{abstract}')],
    'Under the stated initial-certification and report model, paired mobility replays show a quality--cost tradeoff among recruitment policies. At the same payment cap, BTPWR uses 6.55\\% less committed payment and 12.71\\% less physical sampling cost than a credible-first control, with 1.63 percentage points lower completion. A separate synthetic coupling experiment assesses the evidence separation.\n')
text=text.replace('physical sampling cost by 6--15\\% at a measured completion cost of up to 3.08 points, and specify a',
                  'uses less payment and sampling cost than a credible-first control at a measured completion tradeoff, and specify a')
text=text.replace('conduct experiments on mobility replays, hourly\nmeasurements, cross-field checks and contract tests.',
                  'conduct experiments on mobility replays, synthetic evidence controls and contract tests.')
start=text.index('\\section{Experimental Evaluation}')
end=text.index('\\subsection{Blockchain Execution at Experimental Scale}',start)
text=text[:start]+main+text[end:]
start=text.index('Appendix Fig.~\\ref{app-app-fig:coupled} confirms')
end=text.index('The version binds the public history',start)
text=text[:start]+('The corrected synthetic coupling experiments in Appendix~\\ref{app-app:bpqa-deng} quantify the effect of one recovered-value feedback intervention.\n\n')+text[end:]
start=text.index('resulting roster, payment and successor state. A $2\\times2$ factorial')
end=text.index('periodic recertification',start)
text=text[:start]+('resulting roster, payment and successor state. Synthetic controls show the effect of one coupled evidence path; paired comparisons expose a quality--cost tradeoff among paid recruitment policies. The private planner and recovery chain remain a proof prototype. Future work will bind\n')+text[end:]
path.write_text(text,encoding='utf-8')

supp=PAPER/'supplement.tex';s=supp.read_text(encoding='utf-8')
start=s.index('\\subsection{Recovered-Value Coupling Control}')
end=s.index('\\subsection{Progress Model and Paid-Service Planning}',start)
new=r'''
\subsection{Synthetic Selection--Evidence Controls}\label{app:bpqa-deng}\label{app:bpqa-factorial}
We cross the BTPWR planner and a marginal-coverage/reputation selector with
separated or recovered-value-coupled feedback. All four arms share offers,
reports, tasks, recovery and budgets. Coupled missing values are converted
from physical units to the checking scale before scoring, and the planner's
transition history records the feedback actually applied. These synthetic
arms are not full BPQA or Deng implementations. The experiment uses two
mobility patterns, eight seeds and the historical $0.75$ and $1.0$ caps,
giving 32 paired worlds.

\begin{table}[t]\centering\footnotesize
\caption{Selection--evidence factorial: differences in completion points and CE(3).}
\label{tab:bpqa-factorial-app}
\begin{tabular}{lrr}\toprule
Contrast & Completion & CE(3)\\\midrule
Selector change, separated & $+0.03$ & $+0.004$\\
Coupling, BTPWR selector & $-11.88$ & $+0.364$\\
Coupling, alternative selector & $-13.18$ & $+0.388$\\
Interaction & $-1.30$ & $+0.024$\\\bottomrule
\end{tabular}\end{table}
The respective 95\% intervals for the two coupling completion contrasts are
$[-14.80,-8.97]$ and $[-15.10,-11.27]$; for interaction it is
$[-3.29,0.69]$. Intervals average the four pattern/cap strata within each
seed before inference. Coupling is a controlled intervention rather than an
estimate of any published system's performance.

'''
s=s[:start]+new+s[end:]
start=s.index('\\section{Progress-Aware Planning: Confirmation and Service Diagnostics}')
end=s.index('\\section{Mapped-Policy Reference and Disclosure Diagnostics}',start)
new2=r'''
\section{Experimental Inputs and Supplementary Results}
\label{app:planning-confirmation}\label{app:new-recruitment-evidence}\label{app:request50-experiments}
The controlled humidity field is the supplied numerical reference; its source
and time metadata could not be verified. The 30-column public history fixes
the kernel family and calibration before the 50 collection rounds. Each
requested region is sampled once per round; a selected worker's complete
actual route contributes physical cost even if its report is rejected.
T-Drive, Foursquare-TKY and Brightkite observation masks retain their observed
mobility clock and training city bounds. Each worker cohort is drawn from the
training top-200 pool; the 24 primary campaigns use seeds 9801--9808 and the
one-factor sweeps use 9701--9703. Primary and sweep seed sets are disjoint,
although they share the historical trajectories and field.

For all methods the default population has $70\%$ true honest workers and
$10\%$ accurately certified workers, sampled from those honest workers.
Their estimated initial reputation is uniformly sampled on $[.95,1]$ and
rounded to $10^{-4}$; all other estimated scores start at $.5$. Intrinsic
reputations are fixed at generation and used for report simulation, not
revealed to the selector. Accurate certification is assumed free; this
assumption does not estimate the cost of building or maintaining a trusted
registry. The privacy sweep changes $\epsilon_1$ only. Certification shares
are relative to the full population, not to the honest subgroup.

\begin{table}[t]\centering\footnotesize
\caption{Result-to-version map.}\label{tab:version-map}
\begin{tabular}{lll}\toprule
Result & Version & Scope\\\midrule
Primary and sweeps & progress-aware & numerical, 50 rounds\\
Synthetic coupling & progress-aware & 32 paired worlds\\
Reference stress below & mapped/Tent & historical initialization\\
Proof lifecycle & finite N4 & four workers/four cells\\
Quorum settlement & contract interface & 100 payout addresses\\\bottomrule
\end{tabular}\end{table}

\subsection{Baseline Accounting and Numerical Choices}
BPQA numeric follows its published max--min coverage/reputation selection,
critical payments, data range check, truth discovery and reputation update.
The implementation uses $\rho=.9$, task weights 10, and a coverage/reputation
scale estimated from an independent calibration seed on the same source pool;
it does not optimize parameters against BTPWR results. A worker's offered
physical route and bid determine recruitment. An accepted bid reserves its
critical payment under the cap; range-invalid reports are not settled, but
their collected routes still incur sampling cost. There is no inter-round
recycling of unspent commitments. Missing reports contribute only observed
residuals. Truth-discovery weights are normalized and a fixed zero-residual
guard is used because the published expression does not specify those
numerical cases. Cryptographic execution and network traffic are not measured
for BPQA, so this is a numerical adaptation, not a full deployed system.

ETBP-P uses the published priority idea with BTPWR's privacy releases,
budgeted charge, feedback and recovery. Greedy-C/R and Random use the same
pipeline; they are traditional recruitment controls rather than full external
systems. The false-acceptance and detection curves apply BTPWR's evaluation
thresholds to all score trajectories; BPQA's decision threshold and score
update differ, so the curves should be read as score comparisons under a
common rule.

\subsection{Evaluation and Uncertainty}\label{app:reviewer-statistics}
Completion counts every requested task and requires relative error below
$10\%$ (exact agreement for zero truth). Answer rate counts finite answers.
For each pair, MAE/RMSE use the intersection of answered requested tasks.
CE($P$) penalizes every unsuccessful task, including finite estimates outside
the tolerance, with $P$; its ranking can change with $P$. The reported
CE(3) is auxiliary, not a calibrated utility.

For a comparison $d$, each of eight seed means averages its three mobility
strata. The conditional 95\% interval is
$\bar d\pm t_{.975,7}s_d/\sqrt{8}$. The scans have three such seed means;
their shaded intervals therefore have considerable uncertainty. Neither the
2,500 requested tasks within a campaign nor the three fixed mobility sources
are treated as independent seed replicates. Different methods generate their
own subsequent state after the same potential input is sampled.

\begin{figure*}[t]\centering
\includegraphics[width=.96\textwidth]{figures/resource_sensitivity.pdf}
\caption{Committed payment and physical sampling cost under budget,
workforce and behavior sweeps. Points average three paired seeds and three
mobility patterns. The band is a seed-level 95\% interval for BTPWR.}
\label{app-fig:review-resource}
\end{figure*}
\begin{figure*}[t]\centering
\includegraphics[width=.93\textwidth]{figures/round_dynamics.pdf}
\caption{Cumulative completion and CE(3), and mean honest and malicious
estimated reputation, over 50 rounds in the default sweep.}
\label{app-fig:review-rounds}
\end{figure*}

'''
s=s[:start]+new2+s[end:]
supp.write_text(s,encoding='utf-8')
print('manuscript and supplement updated from primary_cases.csv')
