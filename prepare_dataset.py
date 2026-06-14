import csv

def prepare_cisi_all(all_path="dataset/CISI.ALL"):
    docs = []
    current_doc = {}
    current_section = None

    with open(all_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I"):
                if current_doc:
                    docs.append(current_doc)
                doc_id = line.split()[1]
                current_doc = {"doc_id": int(doc_id), "title": "", "text": ""}
                current_section = None

            elif line.startswith(".T"):
                current_section = "title"
            elif line.startswith(".W"):
                current_section = "text"
            elif line.startswith(".A") or line.startswith(".X"):
                current_section = None
            else:
                if current_section == "title":
                    current_doc["title"] += (" " + line.strip())
                elif current_section == "text":
                    current_doc["text"] += (" " + line.strip())

    if current_doc:
        docs.append(current_doc)
    return docs


def prepare_cisi_que(que_path="dataset/CISI.QUE"):
    queries = []
    current_query = {}
    current_section = None

    with open(que_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I"):
                if current_query:
                    queries.append(current_query)
                qid = line.split()[1]
                current_query = {"query_id": int(qid), "query_text": ""}
                current_section = None

            elif line.startswith(".W"):
                current_section = "query_text"
            else:
                if current_section == "query_text":
                    current_query["query_text"] += (" " + line.strip())

    if current_query:
        queries.append(current_query)
    return queries


def prepare_cisi_rel(rel_path="dataset/CISI.REL"):
    pairs = []
    with open(rel_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = int(parts[0])
                doc_id = int(parts[1])
                pairs.append({"query_id": qid, "doc_id": doc_id})
    return pairs


def write_csvs(docs, queries, pairs,
               docs_out="cisi_docs.csv",
               queries_out="cisi_queries.csv",
               rel_out="cisi_rel.csv"):

    with open(docs_out, "w", newline="", encoding="utf-8") as f_docs:
        writer = csv.DictWriter(f_docs, fieldnames=["doc_id", "title", "text"])
        writer.writeheader()
        for d in docs:
            writer.writerow(d)

    with open(queries_out, "w", newline="", encoding="utf-8") as f_q:
        writer = csv.DictWriter(f_q, fieldnames=["query_id", "query_text"])
        writer.writeheader()
        for q in queries:
            writer.writerow(q)

    with open(rel_out, "w", newline="", encoding="utf-8") as f_rel:
        writer = csv.DictWriter(f_rel, fieldnames=["query_id", "doc_id"])
        writer.writeheader()
        for p in pairs:
            writer.writerow(p)


def prepare_cisi(all_path="dataset/CISI.ALL",
                 que_path="dataset/CISI.QRY",
                 rel_path="dataset/CISI.REL"):
    # 1) parse all three files
    docs = prepare_cisi_all(all_path)
    queries = prepare_cisi_que(que_path)
    pairs = prepare_cisi_rel(rel_path)

    # 2) write CSVs
    write_csvs(docs, queries, pairs)

    print(f"Wrote {len(docs)} documents, {len(queries)} queries, {len(pairs)} relevance pairs.")


if __name__ == "__main__":
    prepare_cisi()