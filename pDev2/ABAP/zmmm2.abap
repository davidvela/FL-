
*&---------------------------------------------------------------------*
*& Report  ZMMM2
*&        rest ws call 
*
*&---------------------------------------------------------------------*
*&   22.02.2018 -  David Vela - Creation
*&---------------------------------------------------------------------*
REPORT zmmm2.

PARAMETERS p_txt TYPE string DEFAULT 'WORLD'.

START-OF-SELECTION.

  WRITE 'Hello  ' && p_txt.
  NEW-LINE.
*  RETURN.
  DATA: lo_http_client TYPE REF TO if_http_client,
        lo_rest_client TYPE REF TO cl_rest_http_client,
        lo_request     TYPE REF TO if_rest_entity,
        lo_response    TYPE REF TO if_rest_entity,
        
        lv_url         TYPE string,
        lv_body        TYPE string,
        token          TYPE string,
        agreements     TYPE string,
        data           TYPE string,
        location       TYPE string,
        http_status    TYPE        string,
        lv_status      TYPE  i.
  "Create a structure(or deep) that exactly matches your JSON response
*       abap_response  TYPE        zca_serno_ln,


* Create HTTP intance using RFC restination created
* You can directly use the REST service URL as well
  cl_http_client=>create_by_destination(
   EXPORTING
     destination              = 'FIRSTLIGHT'    " Logical destination (specified in function call)
   IMPORTING
     client                   = lo_http_client    " HTTP Client Abstraction
   EXCEPTIONS
     argument_not_found       = 1
     destination_not_found    = 2
     destination_no_authority = 3
     plugin_not_active        = 4
     internal_error           = 5
     OTHERS                   = 6
 ).

* If you are using cl_http_client=>create_by_url use this code to supress and pass your
* HTTP basic authenication
*  lo_http_client->propertytype_logon_popup = lo_http_client->co_disabled.
*  DATA l_username TYPE string.
*  DATA l_password TYPE string.
*  l_username = 'user'.
*  l_password = 'password'.
*  CALL METHOD lo_http_client->authenticate
*    EXPORTING
*      username = l_username
*      password = l_password.

* Create REST client instance

BREAK-POINT.
"url must be /fp/hello for the POST
  CREATE OBJECT lo_rest_client
    EXPORTING
      io_http_client = lo_http_client.

* Set HTTP version
  lo_http_client->request->set_version( if_http_request=>co_protocol_version_1_0 ).
  IF lo_http_client IS BOUND AND lo_rest_client IS BOUND.
*    DATA(id) = 'XYZ'.
*    CONCATENATE '/agreements/' id INTO lv_url.
*    lv_url = '/123'.

    TRY.
* Set the URI if any
*        cl_http_utility=>set_request_uri(
*          EXPORTING
*            request = lo_http_client->request    " HTTP Framework (iHTTP) HTTP Request
*            uri     = lv_url                     " URI String (in the Form of /path?query-string)
*        ).

* Set request header if any
*     CALL METHOD lo_rest_client->if_rest_client~set_request_header
*       EXPORTING
*         iv_name  = 'auth-token'
*         iv_value = token.

* Set Payload or body ( JSON or XML)
* curl http://localhost:5000/F1L -d "data='{"m":"1","100023":1}'" -X GET
*    # API.payload
*    # {
*    #   "hello": "string",
*    #   "pred": "string"
*    # }

*        perform abap2json.
*        data = '{"m":"1","100023":1}'.
*        data = '-d "data" = ''' &&  '{"m":"1","100023":1}'''.

         data = '{"hello":"' && '{' && |'m':'1','100023':1| && '}","pred": "test" } '.
*        data = '{hello:"123456",pred: "test" } '. "Failed to enconde json object
*        WRITE data.
*
        lv_body = data.
        lo_request = lo_rest_client->if_rest_client~create_request_entity( ).
        lo_request->set_content_type( iv_media_type = if_rest_media_type=>gc_appl_json ).
        lo_request->set_string_data( lv_body ).

* HTTP GET
*        lo_rest_client->if_rest_client~get( ).

* POST
        lo_rest_client->if_rest_resource~post( lo_request ).

* Collect response
        lo_response = lo_rest_client->if_rest_client~get_response_entity( ).
        http_status = lv_status = lo_response->get_header_field( '~status_code' ).
        DATA(reason) = lo_response->get_header_field( '~status_reason' ).
        DATA(content_length) = lo_response->get_header_field( 'content-length' ).
        location = lo_response->get_header_field( 'location' ).
        DATA(content_type) = lo_response->get_header_field( 'content-type' ).
        DATA(json_response) = lo_response->get_string_data( ).
        WRITE json_response.
        RETURN.
* Class to convert the JSON to an ABAP sttructure
*    DATA lr_json_deserializer TYPE REF TO cl_trex_json_deserializer.
*    CREATE OBJECT lr_json_deserializer.
*    lr_json_deserializer->deserialize( EXPORTING json = json_response IMPORTING abap = abap_response ).

      CATCH cx_rest_client_exception.
        WRITE 'exception - REST WS unavailable'.
      CATCH cx_root.
        WRITE 'exception cx_root'.
    ENDTRY.

  ELSE.
    WRITE 'no RFC found'.
  ENDIF.



FORM abap2json.
* ABAP to JSON
  TYPES: BEGIN OF ty_json_req,
           salesorder TYPE string,
           type       TYPE string,
         END OF ty_json_req.

  DATA: json_req TYPE ty_json_req.
  json_req-salesorder = '25252525'.
  json_req-type = 'Direct'.

  DATA lr_json_serializer   TYPE REF TO cl_trex_json_serializer.
  CREATE OBJECT lr_json_serializer EXPORTING data = json_req.

  lr_json_serializer->serialize( ).
  lv_body = lr_json_serializer->get_data( ).

* Or you can use this class as well
* lv_body = /ui2/cl_json=>serialize( data = json_req ).
* Converted JSON should look like this
* lv_body = '{ "salesorder":"25252525", "type":"Direct"}'.

ENDFORM.

FORM http_call.

*  cl_http_client=>create_by_url(
*        EXPORTING
*          url = lv_url
*        IMPORTING
*           client = lo_http_client
*        EXCEPTIONS
*        argument_not_found = 1
*        plugin_not_active = 2
*        internal_error    = 3
*        OTHERS            = 4 ).
*        IF sy-subrc <> 0.
*          RETURN.
*        ENDIF.

*CALL FUNCTION 'SCMS_STRING_TO_XSTRING'
*EXPORTING
*text   = lv_payload
*IMPORTING
*buffer = lv_payload_x.
**
ENDFORM.