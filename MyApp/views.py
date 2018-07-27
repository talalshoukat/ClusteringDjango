
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
from MyApp.read_and_clean_documents import *
from MyApp.text_processing import *
from MyApp.clustering_functions import *
from MyApp.plot import *
#from read_and_clean_documents import *
sys.setrecursionlimit(2900)


DATA_FOLDER = "../data/"
HTML_DATA_FOLDER = "../data/gap-html/"
# Create your views here.
@api_view(["POST"])
def Train(request):
    try:
        inc = read_incident_numbers();
        incidents = []
        content_as_str = []
        for incident in request.data['incidents']:
            incidents.append(incident.get('IncidentId'))
            content_as_str.append(incident.get('Description'))
        #cleaned_content_as_list, cleaned_content_as_str) = \
        #   read_from_cleaned_file('cleaned_descriptions_updated.txt')

        #(frequent_words_removed_content_as_list, frequent_words_removed_content_as_str) = \
        #   read_from_cleaned_file('freq_words_removed_descriptions.txt')

        # (book_names, authors) = read_authors_book_names()
        (similarity_matrix, tfidf_matrix, tfidf_vectorizer) = get_similarity_matrix(content_as_str)

        [assignemnts, clusters] = ward_dendogram(similarity_matrix, incidents)
        id,il=save_json_hierarchy(clusters, incidents, content_as_str)
        #tree generated using incidents is converted into json object
        json.dumps(il, default=ComplexHandler)
        save_ward_hierarchy(clusters, incidents, content_as_str)
        [assignemnts, clusters] = single_dendogram(similarity_matrix, incidents)
        save_single_hierarchy(clusters, incidents, content_as_str)
        [assignemnts, clusters] = complete_dendogram(similarity_matrix, incidents)
        save_complete_hierarchy(clusters, incidents, content_as_str)
        [assignemnts, clusters] = average_dendogram(similarity_matrix, incidents)
        save_average_hierarchy(clusters, incidents, content_as_str)

        #save_hierarchy(clusters, incidents, content_as_str)
        return JsonResponse("Your cluster have been trained", safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)