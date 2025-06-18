def search(df, indexes, model, user_input, k=5):
    index_title, index_content, index_tag = indexes
    query_vec = model.encode(user_input, convert_to_tensor=True)
    query_vec = query_vec / query_vec.norm()
    q = query_vec.cpu().numpy().astype("float32").reshape(1, -1)

    D_title, I_title = index_title.search(q, k)
    D_content, I_content = index_content.search(q, k)
    D_tag, I_tag = index_tag.search(q, k)

    weight_title = 1.0
    weight_content = 2.0
    weight_tag = 0.3

    score_dict = {}

    for rank in range(k):
        idx_t = I_title[0, rank]
        score_dict[idx_t] = score_dict.get(idx_t, 0) + D_title[0, rank] * weight_title

        idx_c = I_content[0, rank]
        score_dict[idx_c] = score_dict.get(idx_c, 0) + D_content[0, rank] * weight_content

        idx_g = I_tag[0, rank]
        score_dict[idx_g] = score_dict.get(idx_g, 0) + D_tag[0, rank] * weight_tag

    top_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    threshold = 0.5
    filtered = [(idx, sc) for idx, sc in top_results if sc >= threshold]

    keyword = user_input.lower()
    filtered = [
        (idx, sc) for idx, sc in filtered
        if (keyword in df.iloc[idx]["タグ"].lower()
            or keyword in df.iloc[idx]["タイトル"].lower()
            or keyword in df.iloc[idx]["内容"].lower())
    ]

    return filtered
