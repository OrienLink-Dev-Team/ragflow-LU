#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime
import json
import traceback

from flask import request
from flask_login import login_required, current_user
from elasticsearch_dsl import Q

from api.db.services.dialog_service import keyword_extraction
from rag.app.qa import rmPrefix, beAdoc
from rag.nlp import search, rag_tokenizer
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils import rmSpace
from api.db import LLMType, ParserType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.db.services.document_service import DocumentService
from api.settings import RetCode, retrievaler, kg_retrievaler
from api.utils.api_utils import get_json_result
import hashlib
import re
import requests


@manager.route('/list', methods=['POST'])
@login_required
@validate_request("doc_id")
def list_chunk():
    req = request.json
    doc_id = req["doc_id"]
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req.get("keywords", "")
    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        query = {
            "doc_ids": [doc_id], "page": page, "size": size, "question": question, "sort": True
        }
        if "available_int" in req:
            query["available_int"] = int(req["available_int"])
        sres = retrievaler.search(query, search.index_name(tenant_id), highlight=True)
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
        for id in sres.ids:
            d = {
                "chunk_id": id,
                "content_with_weight": rmSpace(sres.highlight[id]) if question and id in sres.highlight else sres.field[
                    id].get(
                    "content_with_weight", ""),
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": sres.field[id]["docnm_kwd"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "img_id": sres.field[id].get("img_id", ""),
                "available_int": sres.field[id].get("available_int", 1),
                "positions": sres.field[id].get("position_int", "").split("\t")
            }
            if len(d["positions"]) % 5 == 0:
                poss = []
                for i in range(0, len(d["positions"]), 5):
                    poss.append([float(d["positions"][i]), float(d["positions"][i + 1]), float(d["positions"][i + 2]),
                                 float(d["positions"][i + 3]), float(d["positions"][i + 4])])
                d["positions"] = poss
            res["chunks"].append(d)
        return get_json_result(data=res)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/get', methods=['GET'])
@login_required
def get():
    chunk_id = request.args["chunk_id"]
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")
        res = ELASTICSEARCH.get(
            chunk_id, search.index_name(
                tenants[0].tenant_id))
        if not res.get("found"):
            return server_error_response("Chunk not found")
        id = res["_id"]
        res = res["_source"]
        res["chunk_id"] = id
        k = []
        for n in res.keys():
            if re.search(r"(_vec$|_sm_|_tks|_ltks)", n):
                k.append(n)
        for n in k:
            del res[n]

        return get_json_result(data=res)
    except Exception as e:
        if str(e).find("NotFoundError") >= 0:
            return get_json_result(data=False, retmsg=f'Chunk not found!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/set', methods=['POST'])
@login_required
@validate_request("doc_id", "chunk_id", "content_with_weight",
                  "important_kwd")
def set():
    req = request.json
    d = {
        "id": req["chunk_id"],
        "content_with_weight": req["content_with_weight"]}
    d["content_ltks"] = rag_tokenizer.tokenize(req["content_with_weight"])
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["important_kwd"] = req["important_kwd"]
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(req["important_kwd"]))
    if "available_int" in req:
        d["available_int"] = req["available_int"]

    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embd_id)

        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        if doc.parser_id == ParserType.QA:
            arr = [
                t for t in re.split(
                    r"[\n\t]",
                    req["content_with_weight"]) if len(t) > 1]
            if len(arr) != 2:
                return get_data_error_result(
                    retmsg="Q&A must be separated by TAB/ENTER key.")
            q, a = rmPrefix(arr[0]), rmPrefix(arr[1])
            d = beAdoc(d, arr[0], arr[1], not any(
                [rag_tokenizer.is_chinese(t) for t in q + a]))

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"]])
        v = 0.1 * v[0] + 0.9 * v[1] if doc.parser_id != ParserType.QA else v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        ELASTICSEARCH.upsert([d], search.index_name(tenant_id))
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/switch', methods=['POST'])
@login_required
@validate_request("chunk_ids", "available_int", "doc_id")
def switch():
    req = request.json
    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")
        if not ELASTICSEARCH.upsert([{"id": i, "available_int": int(req["available_int"])} for i in req["chunk_ids"]],
                                    search.index_name(tenant_id)):
            return get_data_error_result(retmsg="Index updating failure")
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@login_required
@validate_request("chunk_ids", "doc_id")
def rm():
    req = request.json
    try:
        if not ELASTICSEARCH.deleteByQuery(
                Q("ids", values=req["chunk_ids"]), search.index_name(current_user.id)):
            return get_data_error_result(retmsg="Index updating failure")
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        deleted_chunk_ids = req["chunk_ids"]
        chunk_number = len(deleted_chunk_ids)
        DocumentService.decrement_chunk_num(doc.id, doc.kb_id, 1, chunk_number, 0)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/create', methods=['POST'])
@login_required
@validate_request("doc_id", "content_with_weight")
def create():
    req = request.json
    md5 = hashlib.md5()
    md5.update((req["content_with_weight"] + req["doc_id"]).encode("utf-8"))
    chunck_id = md5.hexdigest()
    d = {"id": chunck_id, "content_ltks": rag_tokenizer.tokenize(req["content_with_weight"]),
         "content_with_weight": req["content_with_weight"]}
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["important_kwd"] = req.get("important_kwd", [])
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(req.get("important_kwd", [])))
    d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.datetime.now().timestamp()

    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        d["kb_id"] = [doc.kb_id]
        d["docnm_kwd"] = doc.name
        d["doc_id"] = doc.id

        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING.value, embd_id)

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"]])
        v = 0.1 * v[0] + 0.9 * v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        ELASTICSEARCH.upsert([d], search.index_name(tenant_id))

        DocumentService.increment_chunk_num(
            doc.id, doc.kb_id, c, 1, 0)
        return get_json_result(data={"chunk_id": chunck_id})
    except Exception as e:
        return server_error_response(e)


@manager.route('/retrieval_test', methods=['POST'])
@login_required
@validate_request("kb_id", "question")
def retrieval_test():
    req = request.json
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    self_rag = req.get("self_rag", False)
    question = req["question"]
    kb_id = req["kb_id"]
    if isinstance(kb_id, str): kb_id = [kb_id]
    doc_ids = req.get("doc_ids", [])
    similarity_threshold = float(req.get("similarity_threshold", 0.0))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    top = int(req.get("top_k", 1024))

    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        for kid in kb_id:
            for tenant in tenants:
                if KnowledgebaseService.query(
                        tenant_id=tenant.tenant_id, id=kid):
                    break
            else:
                return get_json_result(
                    data=False, retmsg=f'Only owner of knowledgebase authorized for this operation.',
                    retcode=RetCode.OPERATING_ERROR)

        e, kb = KnowledgebaseService.get_by_id(kb_id[0])
        if not e:
            return get_data_error_result(retmsg="Knowledgebase not found!")

        embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)

        rerank_mdl = None
        if req.get("rerank_id"):
            rerank_mdl = LLMBundle(kb.tenant_id, LLMType.RERANK.value, llm_name=req["rerank_id"])

        if req.get("keyword", False):
            chat_mdl = LLMBundle(kb.tenant_id, LLMType.CHAT)
            question += keyword_extraction(chat_mdl, question)

        retr = retrievaler if kb.parser_id != ParserType.KG else kg_retrievaler
        ranks = retr.retrieval(question, embd_mdl, kb.tenant_id, kb_id, page, size,
                               similarity_threshold, vector_similarity_weight, top,
                               doc_ids, rerank_mdl=rerank_mdl, highlight=req.get("highlight"))
        for c in ranks["chunks"]:
            if "vector" in c:
                del c["vector"]
        if self_rag:
            filtered_chunks = []
            for id, chunk in enumerate(ranks["chunks"]):
                try:
                    if id < top:
                        grade_node_response = grade_node(question, chunk["content_with_weight"])
                        grade_result_json = extract_json_list(grade_node_response)
                        if grade_result_json["结果"]:
                            filtered_chunks.append(ranks["chunks"][id])
                    else:
                        break
                except Exception as e:
                    print(f"Self-RAG Error: {e}")
                    continue
            ranks["chunks"] = filtered_chunks

        return get_json_result(data=ranks)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found! Check the chunk status please!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/knowledge_graph', methods=['GET'])
@login_required
def knowledge_graph():
    doc_id = request.args["doc_id"]
    req = {
        "doc_ids":[doc_id],
        "knowledge_graph_kwd": ["graph", "mind_map"]
    }
    tenant_id = DocumentService.get_tenant_id(doc_id)
    sres = retrievaler.search(req, search.index_name(tenant_id))
    obj = {"graph": {}, "mind_map": {}}
    for id in sres.ids[:2]:
        ty = sres.field[id]["knowledge_graph_kwd"]
        try:
            content_json = json.loads(sres.field[id]["content_with_weight"])
        except Exception as e:
            continue

        if ty == 'mind_map':
            node_dict = {}

            def repeat_deal(content_json, node_dict):
                if 'id' in content_json:
                    if content_json['id'] in node_dict:
                        node_name = content_json['id']
                        content_json['id'] += f"({node_dict[content_json['id']]})"
                        node_dict[node_name] += 1
                    else:
                        node_dict[content_json['id']] = 1
                if 'children' in content_json and content_json['children']:
                    for item in content_json['children']:
                        repeat_deal(item, node_dict)

            repeat_deal(content_json, node_dict)

        obj[ty] = content_json

    return get_json_result(data=obj)

def grade_node(query, chunk_content):
    system_prompt = f"""
    以下是问题:
    {query}
    以上是问题。
    以下是信息：
    {chunk_content}
    以上是信息。
    请返回 JSON 对象，包含两个键：结果和原因
    当信息和问题相关或信息包含问题的内容，将结果的键值设置为 true; 当信息和问题完全无关，将结果的键值设置为 false
    原因的键值是对应的原因
    """
    num_ctx = len(system_prompt) if len(system_prompt) < 8000 else 8000
    response = completion_generate(
        model = "qwen2.5:72b",
        base_url="http://10.5.8.11:11434",
        prompt=system_prompt,
        options={
            "temperature":0.8,
            "seed":47,
            "format": "json",
            "num_ctx": num_ctx
        }
    )
    return response

def extract_json_list(res):
    try:
        json_block = res.split("```json")[-1]
        json_block = json_block.split("```")[0]
        return json.loads(json_block)
    except Exception as e:
        raise RuntimeError(f"Extract error : {e}")

def completion_generate(model:str, base_url:str, prompt:str, stream:bool=False, options: dict = {"temperature": 0.8, "seed": 47}) -> str:
    json_param = {
        "model": model,
        "stream": stream,
        "prompt": prompt,
        "options": options,
    }
    try:
        # A100 上 ollama API 是 localhost:11434
        for i in range(5):
            response = requests.post(base_url+"/api/generate", json=json_param, stream=stream)
            if response.status_code == 200:

                if stream:
                    combine_content = ""
                    for stream_chunk in response.iter_content(chunk_size=4096):
                        try:
                            stream_chunk_json = json.loads(stream_chunk.decode('utf-8'))
                            print(stream_chunk_json["message"]["content"])
                            combine_content += stream_chunk_json["message"]["content"]
                        except Exception as e:
                            print(f"completion chat stream error: {e}")
                            continue
                    return combine_content
                else:
                    return json.loads(response.text)["response"]
            else:
                continue
    except Exception as e:
        raise ConnectionError(f"llm chat error: {e}")