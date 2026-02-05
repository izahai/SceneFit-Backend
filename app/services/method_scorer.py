from typing import List, Dict, Any
import json

def score_methods(methods: List[str],
				  participant_responses: List[Dict[str, Any]],
				  *,
				  alpha: float = 0.6,
				  num_outfits: int = 5) -> Dict[str, Any]:
	# sanitize alpha
	if alpha < 0.0:
		alpha = 0.0
	if alpha > 1.0:
		alpha = 1.0

	# initialize structures
	total_participants = len(participant_responses)
	per_method = {}
	for m in methods:
		per_method[m] = {
			'first_stage_counts': {},  # outfit_index -> count
			'view_counts': {i: 0 for i in range(num_outfits)},  # outfit_index -> total view clicks
			'final_choice_count': 0,
			'mrr_sum': 0.0,  # sum of reciprocal ranks for this method
		}

	# Process each participant. Support multiple input shapes including the
	# Unity payload format (selectedRank is 0-based):
	# {
	#   "participantId": "<generated-by-server>",
	#   "responses": [ {"methodId": "Image Editing", "selectedRank": 0, "viewCounts": [0,1,0,0,0]}, ... ],
	#   "finalWinnerMethodId": "CLIP Model"
	# }
	for resp in participant_responses:
		method_choices = None
		views_map: Dict[str, List[int]] = {}
		if isinstance(resp, dict) and isinstance(resp.get('responses'), (list, tuple)):
			method_choices = {}
			for r in resp.get('responses', []):
				if not isinstance(r, dict):
					continue
				# Don't use truthiness for these fields because selectedRank can be 0.
				mid = r.get('methodId')
				if mid is None:
					mid = r.get('method_id')

				sel = r.get('selectedRank')
				if sel is None:
					sel = r.get('selected_rank')
				if sel is None:
					sel = r.get('selected')

				vcounts = r.get('viewCounts')
				if vcounts is None:
					vcounts = r.get('view_counts')
				if mid is None or sel is None:
					continue
				try:
					sr = int(sel)
				except Exception:
					continue
				# selectedRank is expected 0-based (0..num_outfits-1)
				if 0 <= sr < num_outfits:
					method_choices[str(mid)] = sr
					# Optional: viewCounts for visualization/scoring telemetry
					if isinstance(vcounts, (list, tuple)):
						vals: List[int] = []
						for x in list(vcounts)[:num_outfits]:
							try:
								vals.append(max(0, int(x)))
							except Exception:
								vals.append(0)
							# pad if shorter
						while len(vals) < num_outfits:
							vals.append(0)
						views_map[str(mid)] = vals
				else:
					# out-of-range ranks are ignored (contribute 0)
					continue

		# final choice may be provided under several possible keys; prefer the
		# Unity `finalWinnerMethodId` if present.
		final_choice_method = None
		if isinstance(resp, dict):
			final_choice_method = (
				resp.get('finalWinnerMethodId') or resp.get('final_winner_method_id')
				or resp.get('final_choice_method') or resp.get('final_method') or resp.get('final_choice')
			)

		# If not Unity style, fall back to legacy 'method_choices' key
		if method_choices is None:
			method_choices = resp.get('method_choices')

		# normalize method_choices: allow list (ordered) or dict
		choices_map: Dict[str, int] = {}
		if isinstance(method_choices, dict):
			# Expect keys are method ids and values are chosen indices
			for k, v in method_choices.items():
				try:
					choices_map[str(k)] = int(v)
				except Exception:
					# skip malformed entries
					continue
		elif isinstance(method_choices, list) or isinstance(method_choices, tuple):
			# ordered list corresponding to methods list
			for idx, v in enumerate(method_choices):
				if idx >= len(methods):
					break
				try:
					choices_map[methods[idx]] = int(v)
				except Exception:
					continue
		else:
			# malformed or missing; skip this participant
			continue

		# Count first-stage choices per method
		for m, chosen_idx in choices_map.items():
			if m not in per_method:
				# unknown method id; skip
				continue
			counts = per_method[m]['first_stage_counts']
			counts[chosen_idx] = counts.get(chosen_idx, 0) + 1

			# Aggregate view counts (if provided)
			if m in views_map:
				vc = views_map[m]
				for i, c in enumerate(vc[:num_outfits]):
					per_method[m]['view_counts'][i] = per_method[m]['view_counts'].get(i, 0) + int(c)

			# Stage 1: accumulate reciprocal rank for MRR. If chosen_idx is
			# invalid (e.g. out of range or non-int), treat as no contribution (0).
			try:
				if 0 <= int(chosen_idx) < num_outfits:
					rank = int(chosen_idx) + 1  # convert 0-based index to 1-based rank
					per_method[m]['mrr_sum'] += 1.0 / rank
				else:
					# out of range -> contribute 0
					pass
			except Exception:
				pass

		# Count final choice votes (final_choice_method should be one of methods)
		if final_choice_method and final_choice_method in per_method:
			per_method[final_choice_method]['final_choice_count'] += 1

	# Build output
	out = {'methods': {}, 'summary': {'total_participants': total_participants}}
	ranked = []
	for m in methods:
		entry = per_method[m]
		# first-stage proportions over total participants (if a participant omitted a
		# selection for a method, that participant simply doesn't contribute to that count)
		first_counts = entry['first_stage_counts']
		first_props = {}
		# Note: denominator for first-stage proportions could be total_participants
		# or per-method-present-count. We'll expose both: normalized_by_participants
		# and raw counts. Here we normalize by total participants to make comparisons
		# across methods simple.
		for outfit_idx, c in first_counts.items():
			first_props[str(outfit_idx)] = c / total_participants if total_participants > 0 else 0.0

		final_k = entry['final_choice_count']
		# Stage 2: winner-takes-most WinRate (no confidence intervals)
		winrate = final_k / total_participants if total_participants > 0 else 0.0

		# Stage 1: MRR. Per spec, denominator is total participants. Missing or
		# invalid picks contribute 0.
		mrr = entry.get('mrr_sum', 0.0) / total_participants if total_participants > 0 else 0.0

		final_score = alpha * mrr + (1.0 - alpha) * winrate

		out['methods'][m] = {
			'first_stage_counts': {str(k): v for k, v in sorted(first_counts.items())},
			'first_stage_proportions': first_props,
			'view_counts': {str(k): v for k, v in sorted(entry['view_counts'].items())},
			'view_rate': (sum(entry['view_counts'].values()) / total_participants) if total_participants > 0 else 0.0,
			'stage1_mrr': mrr,
			'final_choice_count': final_k,
			'stage2_winrate': winrate,
			'final_score': final_score,
			'alpha': alpha,
		}
		ranked.append((m, final_score))

	# ranking by final-choice proportion (desc)
	ranked.sort(key=lambda x: x[1], reverse=True)
	out['summary']['ranked_methods'] = [r[0] for r in ranked]

	return out


if __name__ == '__main__':
	# Unity-style smoke test payloads (0-based selectedRank)
	methods = [
		"Image Editing",
		"Vision Language Model",
		"CLIP Model",
		"Asthetic Model",
	]
	participant_payloads = [
		{
			"participantId": "uuid-123",
			"responses": [
				{"methodId": "Image Editing", "selectedRank": 0, "viewCounts": [0, 2, 0, 1, 0]},
				{"methodId": "Vision Language Model", "selectedRank": 2, "viewCounts": [1, 0, 2, 0, 0]},
				{"methodId": "CLIP Model", "selectedRank": 1, "viewCounts": [0, 1, 0, 0, 0]},
				{"methodId": "Asthetic Model", "selectedRank": 3, "viewCounts": [0, 0, 0, 3, 1]},
			],
			"finalWinnerMethodId": "CLIP Model",
		},
		{
			"participantId": "uuid-456",
			"responses": [
				{"methodId": "Image Editing", "selectedRank": 1, "viewCounts": [0, 1, 0, 0, 0]},
				{"methodId": "Vision Language Model", "selectedRank": 0, "viewCounts": [2, 0, 1, 0, 0]},
				{"methodId": "CLIP Model", "selectedRank": 3, "viewCounts": [0, 0, 0, 2, 0]},
				{"methodId": "Asthetic Model", "selectedRank": 2, "viewCounts": [0, 0, 1, 0, 0]},
			],
			"finalWinnerMethodId": "Vision Language Model",
		},
		{
			"participantId": "uuid-789",
			"responses": [
				{"methodId": "Image Editing", "selectedRank": 0, "viewCounts": [1, 0, 0, 0, 0]},
				{"methodId": "Vision Language Model", "selectedRank": 1, "viewCounts": [0, 2, 0, 0, 0]},
				{"methodId": "CLIP Model", "selectedRank": 4, "viewCounts": [0, 0, 0, 0, 1]},
				{"methodId": "Asthetic Model", "selectedRank": 3, "viewCounts": [0, 0, 0, 1, 0]},
			],
			"finalWinnerMethodId": "Image Editing",
		},
	]

	print(json.dumps(score_methods(methods, participant_payloads, alpha=0.6, num_outfits=5), indent=2))

    