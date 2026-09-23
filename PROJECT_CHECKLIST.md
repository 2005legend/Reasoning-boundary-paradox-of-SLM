# 📋 RLVR Research Project - Complete Checklist

**Last Updated:** 2026-09-08  
**Project:** Reasoning Boundary Paradox in SLMs with CB-GRPO

---

## 🎯 Current Status (RIGHT NOW)

| Task | Status | Owner | ETA |
|------|--------|-------|-----|
| **Vanilla GRPO 0.5B** | ✅ DONE | You | Completed |
| **CB-GRPO 0.5B** | ⏳ RUNNING | You | 2-4 hours |
| **Vanilla GRPO 1.5B** | 🔄 NEXT | Friend | 4-6 hours |

---

## 📊 Phase 1: Training Experiments (N1 Analysis)

### ✅ Completed
- [x] **Vanilla GRPO 0.5B** (baseline at small scale)
  - Trained: 800 steps
  - Results: Pass@1=25%, Pass@4=65%
  - Checkpoint: `checkpoints/exp2_n1_vanilla_0.5B/`
  - Status: ✅ DONE

### ⏳ In Progress
- [ ] **CB-GRPO 0.5B** (novel method at small scale)
  - Status: Running smoke test → full 800 steps
  - Owner: You
  - Expected: 2-4 hours total
  - Purpose: Test if capacity balancing reduces boundary shrinkage
  - **Action:** Monitor training, wait for completion

### 🔄 Next Up (Parallel)
- [ ] **Vanilla GRPO 1.5B** (baseline at larger scale)
  - Status: Ready to start
  - Owner: Friend
  - Expected: 4-6 hours
  - File: `07_for_friend/VANILLA_GRPO_1.5B_STANDALONE.txt`
  - **Action:** Send file to friend to start training

### 🔜 Future (Conditional)
- [ ] **CB-GRPO 1.5B** (novel method at larger scale)
  - Status: Not started
  - Condition: Only if CB-GRPO 0.5B shows improvement
  - Owner: You or Friend
  - Expected: 4-6 hours
  - **Action:** Wait for CB-GRPO 0.5B results first

---

## 📈 Phase 2: Evaluation & Metrics

### Must Do (Required for Paper)

#### After CB-GRPO 0.5B Finishes:
- [ ] **Evaluate CB-GRPO 0.5B**
  - [ ] Run Pass@k evaluation (n=32, k=[1,4,16,32])
  - [ ] Compute Pass@k on GSM8K test
  - [ ] Compute Pass@k on GSM8K-platinum
  - [ ] Compute Pass@k on MATH-500
  - [ ] Save results: `results/exp2_n1_cbgrpo_0.5B/pass_at_k.json`
  - **Owner:** You
  - **Time:** 1-2 hours

- [ ] **Compare 0.5B: Vanilla vs CB-GRPO**
  - [ ] Load Vanilla 0.5B results (already have)
  - [ ] Load CB-GRPO 0.5B results (after eval)
  - [ ] Compute Δ Pass@k = Pass@k_cbgrpo - Pass@k_vanilla
  - [ ] Compute shrinkage slope for both
  - [ ] Compute Δ slope = slope_cbgrpo - slope_vanilla
  - [ ] Compute transition matrix (kept_correct, lost_capability, etc.)
  - [ ] **Key Result:** Does CB-GRPO help? (Δ slope should be positive)
  - **Owner:** You
  - **Time:** 30 min

#### After Vanilla 1.5B Finishes:
- [ ] **Evaluate Vanilla 1.5B**
  - [ ] Run Pass@k evaluation (n=32, k=[1,4,16,32])
  - [ ] Compute on all 3 datasets
  - [ ] Save results: `results/exp2_n1_vanilla_1.5B/pass_at_k.json`
  - **Owner:** You or Friend
  - **Time:** 2-3 hours (larger model)

- [ ] **Compare Scales: 0.5B vs 1.5B**
  - [ ] Compare Vanilla 0.5B vs Vanilla 1.5B
  - [ ] Does boundary shrinkage worsen at larger scale?
  - [ ] Compute shrinkage slope for both scales
  - **Key Result:** Scale effect on paradox
  - **Owner:** You
  - **Time:** 30 min

#### If CB-GRPO 1.5B is Trained:
- [ ] **Full 4-way Comparison**
  - [ ] Vanilla 0.5B vs CB-GRPO 0.5B vs Vanilla 1.5B vs CB-GRPO 1.5B
  - [ ] Does CB-GRPO help at both scales?
  - [ ] Is improvement consistent across scales?
  - **Owner:** You
  - **Time:** 1 hour

### Baseline Evaluation (Nice to Have)
- [ ] **Exp0: Base Model Evaluation**
  - [ ] Evaluate Qwen2.5-0.5B-Instruct (base, no training)
  - [ ] Evaluate Qwen2.5-1.5B-Instruct (base, no training)
  - [ ] Just inference, n=32 solutions per problem
  - [ ] Purpose: Compute Δ Pass@k from base model
  - **Owner:** Friend (can do in parallel)
  - **Time:** 1-2 hours each
  - **Priority:** Medium (useful but not critical)

---

## 📊 Phase 3: Visualization & Analysis

### Must Create (For Paper)
- [ ] **Training Curves**
  - [ ] Loss curve over steps (all experiments)
  - [ ] Mean reward curve
  - [ ] Format success rate
  - [ ] Correctness success rate
  - **Time:** 30 min
  - **Tools:** matplotlib/plotly

- [ ] **Pass@k Comparison Plot**
  - [ ] Bar chart: Vanilla vs CB-GRPO at each k
  - [ ] Show both 0.5B and 1.5B if available
  - [ ] Error bars from bootstrap
  - **Time:** 30 min

- [ ] **Shrinkage Slope Visualization**
  - [ ] Plot log(k) vs Δ Pass@k
  - [ ] Show regression lines
  - [ ] Highlight slope differences
  - **Key Figure:** Main result for N1 novelty
  - **Time:** 30 min

- [ ] **Transition Matrix Heatmap**
  - [ ] Kept correct, lost capability, gained capability
  - [ ] Compare Vanilla vs CB-GRPO
  - **Time:** 30 min

### CB-GRPO Specific (N3 Novelty)
- [ ] **Cluster Spend Analysis**
  - [ ] Histogram: Vanilla (uniform) vs CB-GRPO (balanced)
  - [ ] Gini coefficient comparison
  - [ ] Time series: cluster spend evolution
  - [ ] Table: spend distribution statistics
  - **Purpose:** Prove capacity balancing works
  - **Time:** 1 hour

---

## 📝 Phase 4: Paper Writing

### Required Sections

#### Abstract & Introduction
- [ ] **Problem Statement**
  - [ ] Define "reasoning boundary paradox"
  - [ ] Cite existing work on boundary shrinkage
  - **Time:** 1-2 hours

- [ ] **Motivation**
  - [ ] Why SLMs matter (resource constraints)
  - [ ] Why boundary shrinkage is a problem
  - **Time:** 1 hour

#### Related Work (N2 Novelty)
- [ ] **Existing Mitigation Techniques**
  - [ ] O-SELF (online solve-rate filtering)
  - [ ] Static SELF (precomputed filtering)
  - [ ] Adaptive Rollout (variance-based)
  - [ ] Why they don't transfer to SLMs
  - **Time:** 2-3 hours

#### Method (N3 Novelty)
- [ ] **CB-GRPO Algorithm**
  - [ ] Describe cluster-based balancing
  - [ ] EMA spend tracking
  - [ ] Capacity-aware reweighting
  - [ ] Algorithm pseudocode
  - **Time:** 2-3 hours

- [ ] **Implementation Details**
  - [ ] Model: Qwen2.5 (0.5B, 1.5B)
  - [ ] QLoRA 4-bit quantization
  - [ ] Training hyperparameters
  - [ ] Reward function design
  - **Time:** 1 hour

#### Experiments (N1 Results)
- [ ] **Experimental Setup**
  - [ ] Dataset: GSM8K (7,473 problems)
  - [ ] Compute tier: Colab T4 GPU
  - [ ] Training steps: 800 (standard tier)
  - **Time:** 1 hour

- [ ] **Results Section**
  - [ ] Present Pass@k tables
  - [ ] Show shrinkage slope comparison
  - [ ] Present transition matrices
  - [ ] Statistical significance tests
  - **Key Result:** Δ slope improvement
  - **Time:** 2-3 hours

- [ ] **Cluster Analysis**
  - [ ] Show spend distribution
  - [ ] Prove balancing works
  - [ ] Link to improvement
  - **Time:** 1-2 hours

#### Discussion & Conclusion
- [ ] **Findings Summary**
  - [ ] CB-GRPO reduces boundary shrinkage by X%
  - [ ] Works at both 0.5B and 1.5B scales (if tested)
  - [ ] Capacity balancing mechanism effective
  - **Time:** 1 hour

- [ ] **Limitations**
  - [ ] Tested only on GSM8K
  - [ ] Only 2 model sizes
  - [ ] Colab T4 constraints
  - **Time:** 30 min

- [ ] **Future Work**
  - [ ] Test on other datasets (MATH, ARC)
  - [ ] Test larger models (3B, 7B)
  - [ ] Combine with other techniques
  - **Time:** 30 min

---

## 🔧 Phase 5: Code & Artifacts

### Code Repository
- [ ] **Clean Up Code**
  - [ ] Organize into clear structure
  - [ ] Remove deprecated cells
  - [ ] Add documentation
  - **Time:** 2 hours

- [ ] **Create README**
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Reproduce results
  - **Time:** 1 hour

- [ ] **Requirements File**
  - [ ] List all dependencies with versions
  - [ ] Test in fresh environment
  - **Time:** 30 min

### Artifacts for Submission
- [ ] **Trained Models**
  - [ ] Upload checkpoints (if sharing)
  - [ ] HuggingFace Hub or Zenodo
  - **Time:** 1 hour

- [ ] **Results Data**
  - [ ] All evaluation JSONs
  - [ ] Training metrics
  - [ ] Cluster assignments
  - **Time:** 30 min

- [ ] **Figures**
  - [ ] High-res PDFs for paper
  - [ ] Follow conference style guide
  - **Time:** 1 hour

---

## 📅 Timeline Estimate

### Week 1 (Current)
- **Day 1-2:** Training
  - ✅ Vanilla 0.5B (done)
  - ⏳ CB-GRPO 0.5B (running)
  - 🔄 Vanilla 1.5B (friend starts)

- **Day 3:** Evaluation
  - Evaluate all trained models
  - Compute metrics
  - Initial comparison

- **Day 4-5:** Analysis
  - Create visualizations
  - Statistical tests
  - Cluster spend analysis

### Week 2
- **Day 1-3:** Paper Writing
  - Draft all sections
  - Create figures
  - Results tables

- **Day 4-5:** Revision
  - Polish writing
  - Peer review
  - Finalize

### Week 3 (Buffer)
- Final edits
- Submit!

---

## 🎯 Critical Path (Must Complete for Paper)

### Minimum Viable Paper (MVP):
1. ✅ Vanilla 0.5B training (done)
2. ⏳ CB-GRPO 0.5B training (running)
3. ⏳ Evaluate both models
4. ⏳ Compute Δ slope (key metric)
5. ⏳ Create main figures (Pass@k, shrinkage slope)
6. ⏳ Write paper draft

**Minimum for N1 Novelty:** Show CB-GRPO improves over Vanilla at 0.5B

### Enhanced Paper (With 1.5B):
- Add Vanilla 1.5B results
- Show scale effect
- Potentially add CB-GRPO 1.5B
- Stronger N1 claim

---

## 🚨 Blockers & Risks

### Current Blockers: NONE ✅
- CB-GRPO 0.5B training is running
- Friend has Vanilla 1.5B code ready

### Potential Risks:
1. **CB-GRPO doesn't help**
   - Mitigation: Analyze why, still publishable (negative result)
   - Pivot: Focus on scale analysis instead

2. **Colab disconnects**
   - Mitigation: Checkpoints every 200 steps
   - Can resume training

3. **Not enough improvement**
   - Mitigation: Statistical analysis, deeper investigation
   - Alternative: Test different hyperparameters

4. **Time constraints**
   - Mitigation: Focus on MVP (0.5B only)
   - 1.5B can be future work

---

## 📊 Success Metrics

### Paper Acceptance Criteria:
- ✅ **N1 Novelty:** CB-GRPO reduces shrinkage slope by ≥10%
- ✅ **N2 Novelty:** Show existing methods don't work for SLMs
- ✅ **N3 Novelty:** Cluster balancing mechanism analysis
- ✅ **Statistical Significance:** p < 0.05 on improvement
- ✅ **Reproducibility:** Code + checkpoints available

### Strong Paper:
- Δ slope improvement ≥20%
- Works at both 0.5B and 1.5B scales
- Clear cluster spend distribution difference
- Transition matrix shows less capability loss

### Acceptable Paper (MVP):
- Δ slope improvement ≥10%
- Works at 0.5B scale
- Basic cluster analysis
- Promise for future scaling

---

## 🔄 Next Actions (Priority Order)

### Immediate (Next 24 Hours):
1. ⏳ **Monitor CB-GRPO 0.5B training** (you)
   - Check for errors
   - Watch GPU usage
   - Wait for completion (~2-4 hours remaining)

2. 🔄 **Start Vanilla 1.5B training** (friend)
   - Send `VANILLA_GRPO_1.5B_STANDALONE.txt`
   - Friend runs on their Colab
   - Runs in parallel (~4-6 hours)

### After CB-GRPO 0.5B Finishes:
3. **Evaluate CB-GRPO 0.5B** (you)
   - Run Pass@k evaluation
   - Save results

4. **Compare Vanilla vs CB-GRPO at 0.5B** (you)
   - Compute Δ slope
   - **KEY DECISION POINT:** Does it help?

### If CB-GRPO Helps (Δ slope > 0):
5. **Create visualizations** (you)
   - Pass@k comparison
   - Shrinkage slope plot
   - Cluster spend analysis

6. **Start paper writing** (you)
   - While waiting for Vanilla 1.5B to finish
   - Draft method section
   - Draft initial results

### After Vanilla 1.5B Finishes:
7. **Evaluate Vanilla 1.5B** (you or friend)
   - Complete evaluation suite

8. **Decide on CB-GRPO 1.5B** (you)
   - If time permits and CB-GRPO helped at 0.5B
   - Optional: train CB-GRPO 1.5B

9. **Complete paper** (you)
   - Add all results
   - Finalize figures
   - Write discussion

10. **Submit!** 🎉

---

## 📞 Collaboration Checkpoints

### Daily Sync (While Training):
- **What:** Status update on training progress
- **When:** End of each day
- **Who:** You + Friend

### After Each Model Finishes:
- **What:** Share checkpoint paths and metrics
- **When:** Immediately after training completes
- **Who:** You ← Friend (or vice versa)

### Major Decision Points:
1. **After CB-GRPO 0.5B eval:** Does it help?
2. **After Vanilla 1.5B eval:** Train CB-GRPO 1.5B?
3. **After all training:** Start paper writing

---

## ✅ Daily Checklist Template

### Today (While CB-GRPO 0.5B Running):
- [ ] Check CB-GRPO training status (every few hours)
- [ ] Verify checkpoints are saving to Drive
- [ ] Monitor GPU usage (should be steady)
- [ ] Send Vanilla 1.5B file to friend
- [ ] Confirm friend started training

### Tomorrow (After CB-GRPO 0.5B Finishes):
- [ ] Run Pass@k evaluation on CB-GRPO 0.5B
- [ ] Compare with Vanilla 0.5B results
- [ ] Compute key metrics (Δ slope)
- [ ] Create first visualization (Pass@k comparison)
- [ ] Check friend's Vanilla 1.5B progress

### Day After (Analysis Day):
- [ ] Create all visualizations
- [ ] Statistical significance tests
- [ ] Cluster spend analysis
- [ ] Draft method section
- [ ] Wait for Vanilla 1.5B to complete

---

## 🎯 Bottom Line

**Current Focus:**
- ✅ Vanilla 0.5B: Done
- ⏳ CB-GRPO 0.5B: Running (you) - Monitor
- 🔄 Vanilla 1.5B: Starting (friend) - Send file

**Next Up:**
- Evaluate CB-GRPO 0.5B when it finishes
- Compare with Vanilla 0.5B
- Make key decision: Does CB-GRPO help?

**End Goal:**
- Paper showing CB-GRPO reduces boundary shrinkage
- Submitted to conference
- Code + models released

**Timeline:** 2-3 weeks total (1 week training + eval, 1 week writing, 1 week buffer)

---

**Status:** ON TRACK ✅  
**Blockers:** NONE  
**Risk Level:** LOW  
**Confidence:** HIGH 🚀
