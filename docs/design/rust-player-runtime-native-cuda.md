# Native MLP row CUDA storage estimate

Status: scoped pure row-estimator implementation accepted after independent
correctness/conformance READY and fresh57 CPU/static checks. Independent architecture READY on proposal SHA256
`415ffa2915df39fd57c5c576d58eabecaa523031aa235af80ef8fd9c63689272`. Owner: root.
This pure mathematical component supports the required generic device planner in
[device-planning.md](rust-player-runtime-device-planning.md). It does not itself
admit a host/row, acquire authority, set a quota or activate public CUDA mode.
The complete retained-learner/cache/context/phase composition still needs its own
closed design and architecture gate.

## Requirements, responsibilities and interface

Implement a new pure module src/gf_models/rust_player_cuda.py after independent
preimplementation review. Reuse the existing actor_cuda small kernel-storage
primitives where their operands/lifetimes are identical; rebuild raw-input
branches and native normalization/return phases according to the appendix. Do
not call the whole legacy actor estimate and subtract a guessed scalar. Existing
actor estimates, network math, budgets, schemas and legacy modes stay unchanged.
No Torch/native import, device query, filesystem I/O, dependency or allocation
probe belongs in this module.

One immutable NativeMlpRowCudaProfile is compiled from an already validated
ModelConfigV3 or ModelConfigV4 with model_id=mlp, its actual InferenceBindingV2 and
StructuredLayout. Precompute fixed dimensions/branch constants per checkpoint,
without summing parameter shapes on every inference request. The profile has
estimate(*, interval_events: int, candidates: int, input_leaf_bytes: int),
returning an immutable NativeMlpRowCudaEstimate with recipe, input_leaf_bytes,
additional_phases, peak_additional_bytes, peak_phase and peak_bytes. Recipe is
native-mlp-row-cuda-storage-v1. additional_phases are named nonnegative byte
counts, excluding R, weights/positions, learner and context/workspace;
peak_bytes = input_leaf_bytes + peak_additional_bytes. Transfer overlap between
old/new rows belongs to the future whole-host planner, not this single-row value.
These values confer no Source/inference/physical capability and need no new
persisted normative record.

Caller supplies actual native final-dtype input-leaf bytes and actual candidate/
interval lengths, already admitted by the existing native/model contracts.
Count distinct resulting CUDA leaf allocations even if CPU values duplicate;
count slicing/reshape/detach aliases once. Validate integration assumptions at
this boundary without revalidating observations/tokens or inventing new model
limits. CPU callers may calculate the CUDA storage geometry without initializing
CUDA; this is a planned CUDA estimate, not measured CPU/CUDA allocation.

All layout categories and model branches use the actual checkpoint metadata;
heterogeneous policies do not inherit the learner's geometry. Estimate actual
zero events separately from positive events. A worst-config caller must consider
both zero and maximum-positive branches; monotonicity across that branch change
is not assumed. For candidate chunks use the last full and last partial chunk,
not an O(candidate_count) loop. The hot-path calculation should be fixed-size
arithmetic over precomputed constants, with no tensor/array construction.

## Verification and limits

Tests belong in a new tests/test_rust_player_cuda.py after ownership assignment.
Use TDD and independent allocation/lifetime expectations, not tests that merely
repeat a copied formula. Cover all raw aliases, new inverted masks, zero/positive
events with hand attention, identity state, direct/sum/mean/max pooling, partial
chunks, retained scores through critic/finite checks, output probability/CPU
transfer, heterogeneous row geometry, and R counted exactly once. Reuse documented
kernel assumptions; optional tiny CPU dispatch checks must distinguish the CPU
GRU implementation from the CUDA formula. No GPU/training/performance campaign
is authorized by this module design. Actual planned CUDA observations later test
the model without making it a hard allocator/physical bound.

Existing configured backend workspace assumptions remain explicit. This module
does not cover learner restore/update minima, cache publication overlap or
heterogeneous workspace policy. Those remain whole-host requirements before
public activation. Source-derived estimates and actual allocated/reserved/PSS
measurements must be reported separately.

## Source-derived allocation appendix for independent review

The following advisor table is proposed evidence, not a completed implementation
or an unreviewed production restriction. Root adopts it as the candidate
mathematical decomposition for the narrow architecture review.

# Read-only advisor native tensor-storage proposal (NOT yet approved code)

/root/rust_player_gpu_advice returned this source-derived table. Root must turn it
into a bounded design and obtain independent architecture review before generic
CUDA guard implementation/public activation. No source files/GPU/training changed.
A7element CPU TorchDispatch trace of pinnedTorch2.14 isfinite was run; no CUDA
query/allocation/peak evidence. Model3 restore minima NOTDERIVED (verifierongoing).

Scope actual shared_numeric.score_numeric_row -> structured_network frame/candidate
operands; rust_player.py evaluate/provisional_sample. All terms below exclude
inputR, cache/learner/network parameters+positions and configuredsharedcontext/
workspace. Tensor storage estimate, NOT allocatorreserved/physicalbound.
R=sumACTUALDISTINCTcudaInputleaves. Native separateleaves count separatelyevenequal
values; rawstate/card/event/history/candidatearraysallaliasR. Views/[None]/flatten/
detach/referenceSlices add0. Stateidentity state aliasesstate_input; hand_cards
aliases cards whennonattention; event aliases GRUhidden.
rowPeak=R+max(frame,candidate,output phases); transferfirstRnew, laterRold+Rnew,
not2Rcurrent acrossheterogeneousgeometry. Currentevaluate moveslogs/valueCPUbefore
nextrow, butretains inputdictuntilnextRHS assignment (actualB>1 overlap).

Primitives: rawFP32x bytes _float_tensor peakADDITIONAL3x (sign/abs/log/product),
rawxalreadyR. Onehotnrows categorywidthswi: F=4n*sum(wi);
OH=max(2F,max_i[4n*sum_{j<i}(wj)+12nwi]). Priorfloatcolumns+currentint64/float,
thenlist+cat. Existing_onehot_peak-raw isconservative; donotguessglobal subtraction.

Framevariables: Xs rawstate numericbytes, Fs expandedstatecategoryfloatbytes,
Oexpandedopponentfloat,Scompletedstateinput, A=S+stateOutput(0foridentity),
Xc rawcard numeric,N=2Xc completednumeric/presentcat,D=definition+abilityembeddings,
Fc suffixcategoryfloat,Icardinput,Cencodedcards,Tmembershiponehot,J=4*18 order,
ACactorcardconcat,m=18 newlyinvertedboolmask, P=A+N+I+C+T+J+AC.
Operation additional live bytes (network273..292):
statefloat3Xs; statecategories Xs+OHstate; opponent Xs+Fs+3O;
stateconcat Xs+Fs+O+S; repeatedopponentmask S+O;
stateencoder S+existing_state_mlp_scratch (identity0);
cardnumeric A+3Xc;
cardembed/suffix/cat A+N+max(D+OHsuffix,D+Fc+I);
cardencode/activate/mask A+N+I+2C+m;
membershiponehot/mask A+N+I+C+max(3T,2T+m);
order A+N+I+C+T+3J; actorcardcat P+4*18.
Raw mask inR, inverted~mask NEW; cannotremoveeverymaskterm.

Handextra beyondP:
direct max(OHallcards,2H+m), Hcompletedhand.
nonattention max(existingencoderScratch,2L+m,poolingPeak), Llatent.
attention reuse_attention_scratch pluspriorlayeroutput/KV duringnewRHS;
encoderafterattention retainfinalattentionC+KV2C plusencoder/pool expression.
Poolpeak includesL: sum L+H; mean L+max(H+16,2H+8);
max max(2L+m,2L+3H,L+3H+1), values+int64indices oroldhand/zeros/where.
Fhand=P+attentionOutput+attentionKV+L+H (directattention/L0).

Positiveevent: Xe rawintervalFP32, Fe expandedcategories,Einput=2Xe+Fe;
eventPack=max(3Xe,Xe+OHevents,Xe+Fe+Einput);
eventPeak=Fhand+max(eventPack,Einput+EXISTINGCUDAevent_gruScratch).
ReusableactorCudaGRU121..128 sameGRU/inputexternal; CPUfullgatesdifferentfromCUDA,
CPUtraceisNOT CUDApeakproof.
Fpositive=Fhand-attentionKV+Einput+eventSequenceOutput+fullHiddenOutput.
'_'retainsoldattentionKV THROUGH GRURHS andisreplacedonlyonassignment.
ZEROevents: Fzero=Fhand+4*eventHidden. NoGRUassignment, '_' STILLownsattentionKV.
ThisbranchmustNOTmechanicallysubtracthandKV likeoldpositiveeventactorformula.
ForbaseFb, historyXh, completedFrame, contextHctx:
projectionPeak=Fb+max(3Xh,Xh+Frame,Frame+2Hctx,
 Frame+Hctx+existingTemporalDenseScratch whenpresent).
Afterframefunction returns ONLYcontext+actorcards survive: U=Hctx+AC.
No separatetemporalcache/stepallocation forMLP scorer.

Candidate: nactualcandidates,aactorFirstWidth,V=4a sharedprojection,
Kkindonehot,Tg targetonehot,X=16n numeric4float,F=K+Tg+2X candidatefeature.
featurePeak=U+V+max(3K,K+3Tg,K+Tg+3X,K+Tg+X+F).
CardProjection reuseexistingcandidate_projection (operandcopy/einsum+unpadded/
pad/cat NEW). positions retain8*maxPositions. refsVIEWofR: no refalloc/newoldcopy.
Bscore=EXISTING scoring_fixed withNO referencecharge. chunkPeak(j)=Bscore+
4*previousCandidateCount(j)+existingChunkScratch(rows(j)); concatPeak=Bscore+8n.
Lastfullchunk+finalpartial suffice(maxearlierequalchunksfewerlogits).

REPLACE oldactor normalization/critic phases:
logsoftmax U+8n; critic U+8n+existingCriticMlpScratch;
finite U+8n+4+5n; sampling/logs.exp+probCPUtransfer 8n+4;
logsCPUreturn4n+4; valueCPUreturn4.
ScoresSURVIVESlogsoftmax+critic+finite(shared_numeric60..61), retirewithcontext/cards
onreturn. ProbabilityCUDAtemp survivesduringprovisional_sample, CPUmultinomial/
validation addNOcuda. Finite5n explicitFP32abs+bool compare; CPUtraceabs→ne→eq→mul→all
andstaticCompositeImplicitAutogradregistration supportsdecomposition, NOTmeasured
CUDApeak. All answerCPUaccumulation notcuda, previouslogs/valueCPUatnextH2D.

Wholehost=maxREACHABLEphase [oneapplicablecontext/workspace +actualretainedlearner
+actualDISTINCTcachestorage +phaseadditional]. ExistinglearnerestimateALREADYhas
context, donotaddtwice. EachnetworkPi+Qi fromOWNModel3/4metadata, Qi nonpersistent
positions. Equalvalues!=alias; learnerandimmutableinference separateactualnetwork
storage. Countconstructingpubbeforeinsertion, incomingCPUbanksnotwholeGPUclone.
Heterogeneouscheckpointworkspaceallowances mustbindapplicablephaseexplicitly,
notlearner'spossiblysmaller orsumpernetworkcontexts. Rootneedsdecision.
Model3 fresh/stagedrestoreseparatederivationstillrequiredbeforegenericguard.

Required boundedverification: sourceformulabranchreview, zero/positiveevents+
attention,identitystate,direct/pooledhand,partialchunks,outputlifetimes;
heterogeneouscheckpointgeometry/puboverlap/B1/B>1transferpairs; actualplannedCUDA
measurementscheckmodel, noextraformalcampaignneeded. No speculativecoefficient,
capacity7 orC/Wmultiplier/actorTransient+R accepted.

## Scoped implementation acceptance

The immutable profile and read-only E/n/R results pass both independent reviews
with no findings. Root and the correctness reviewer freshly run57 CPU cases
(33 new plus24 existing pure configuration regressions); scoped mypy/Ruff pass.
Module SHA256 is `31de9f939176b43714f191687b28d8662bd9e2b79379f16f2bf7f9c42589bd6e`,
tests `db934745c1880738db2dffe532429a0d5af91fcff0e4bd728c2eb594c8101088`.
Import/estimate does not load Torch, NumPy or gf_native. A bounded CPU arithmetic
microcheck reports about2.3us per estimate at n1/65536; no GPU or end-to-end claim.
Kernel/workspace assumptions remain tensor-storage estimates. Whole-host device
accounting, operation allowances, actual GPU validation, public activation and
all Task6/7 evidence remain separate required work.
