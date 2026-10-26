# Proprietary factorial protocol provenance

This record preserves the execution chronology previously typeset in Document S1 Table S16. The publication now describes the implemented protocol in Table S16 and retains the analysis-timing qualifications in Methods; exact timestamps, commit IDs, abandoned specifications and cache-integrity evidence remain here. No trial data or analysis results were changed when shortening the table. Original repository history is unchanged.

## Detailed archived record

The following is the original LaTeX record, preserved verbatim for reproducibility. Times are Tallinn local time (UTC+3). Start times establish execution order, not result blinding.

```latex
\section{Factorial protocol history and deviations}

The factorial plan is \texttt{docs/factorial\_preregistration.md} in the code repository, a timestamped repository plan rather than an external registration. Table~S16 summarizes plan versions, reported run timing and deviations; per-cell manifests record start times and code commits. Times are Tallinn local time (UTC+3). Three Gemini pilot cells were retained. At GPT's first amendment (8 September, 12:46), 45 of 48 Gemini cells had started; all 48 had started by its second amendment (14:55). These start times establish campaign order, not result-access or blinding. An initial GPT caption-pilot manifest starts at 11:44, before both amendments, so a blanket claim that every amendment preceded every affected call is inappropriate. The completed GPT reasoning cells followed the amendments. The 303 pilot captions were retained, not regenerated: the final caption cache begins with the pilot's 854 records unchanged, each of the 2,504 images has exactly one successful caption, and the pilot and final caption manifests agree on prompt version, system-prompt hash, seed and output budget. These captions were written under the plan then in force, in which GPT wrote its own captions, which is also the condition the second amendment restored.

\begin{table}[ht]
\centering
\caption{Factorial plan versions, run timing and deviations from the original plan (commit \texttt{46dfbb7}).}
\small
\begin{tabularx}{\textwidth}{p{0.24\textwidth}Y}
\toprule
Item & Record \\
\midrule
Original plan & \texttt{46dfbb7}, 7 Sep 13:58. Ten aliases (eight vision, two text-only); primary analysis a logistic mixed model on trial correctness with item random intercepts; 10,000 base-problem resamples; caption fidelity, factorial human--machine alignment and wrong-cue susceptibility as secondary analyses; no replacement models after the pilot. \\
Caption prompt & Revised to v3 at \texttt{7a482fc} (7 Sep 16:05) after a probe showed invented second shapes. The local Gemini v3 cache manifest starts at 17:19 and contains 2,504 successful v3 captions; older v1/v2 captions remain in the cache but are excluded by version filtering. \\
Pilot cells & Gemini, no cue, greedy run: pixels, own captions and benchmark descriptions, started 16:35--16:50 under \texttt{7a482fc}. Folded into the full-run tree and included among the 48 reported Gemini cells. \\
Panel amendment & \texttt{0b64a54}, 7 Sep 18:17: panel restricted to Gemini~3.5 Flash, GPT~5.5 and Qwen3-VL:30b, the aliases with archived paired coverage. No cell of a removed model had been run. \\
Gemini arm & Remaining 45 cell starts span 7 Sep 19:08 to 8 Sep 13:32, after the panel amendment. Gemini serves as the reference captioner, so \texttt{caption\_strong} would duplicate its own captions (48 cells in total). \\
GPT~5.5 amendments & \texttt{5717621} (8 Sep 12:46): use Gemini's captions instead of its own. \texttt{f291f65} (8 Sep 14:55): restore its own captions and keep Gemini's as \texttt{caption\_strong}; two sampled runs instead of three after the provider rejected 551 of the first 854 caption calls for rate limits. The greedy run is absent as originally planned, because the provider fixes temperature. \\
GPT~5.5 arm & Caption pilot, manifest start 8 Sep 11:44, before both GPT amendments: 854 calls, 303 captions and 551 rate-limit rejections. The 303 captions were retained in the final cache unchanged. The remaining 2,201 captions (the 551 rejected images and 1,650 not yet attempted) from 15:08; all 30 reasoning cells 9--17 Sep under \texttt{eca1c26}. \\
Open-weight arm & Qwen3-VL:30b stopped in the caption stage (453 of 2,504 captions in 44.3 GPU-hours). Gemma4:31b substituted on 14 Sep (\texttt{6305c13}), contrary to the no-replacement rule; it also did not complete. Both campaigns ran with model-default reasoning enabled because the backend did not transmit the disabled thinking setting (fixed in \texttt{c284932}). No open-weight data from this design are reported; a separately planned replication (\texttt{ec3efd2}) is described in Section~\ref{sec:owreplication}. \\
Primary analysis & Original mixed-model specification replaced by post hoc GEE correctness contrasts and a variational-Bayes item-intercept sensitivity (Section~\ref{sec:robustness}). Main summaries are run-averaged accuracy, $d'$ and $c$, with family-stratified base-problem bootstrap intervals. \\
Resamples & 4,000 for the recovered shares; 2,000 for the cue-overlap contrasts. \\
Secondary analyses & Caption fidelity completed after the results were known (Section~\ref{sec:robustness}); factorial human--machine alignment and wrong-cue susceptibility were not performed. \\
Added analyses & The recovered share on the $d'$ scale, operational denominators (Table~S13), cue-overlap sensitivity and the renderer-faithful description check (Table~S12) were added after the completed runs. A boundary-stratified analysis was withdrawn after its turn convention was found to be reversed. \\
\bottomrule
\end{tabularx}
\end{table}


```
