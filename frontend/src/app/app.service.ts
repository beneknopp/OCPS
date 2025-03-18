import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { MessageService } from './message.service';


@Injectable({
  providedIn: 'root'
})
export class AppService {
  
  private backendUrl = 'http://localhost:5000/';
  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
    })
  };

  constructor(
    private http: HttpClient,
    private messageService: MessageService
  ) { }

  getPing(): Observable<String> {
    return this.http.get<String>(this.backendUrl, { headers: new HttpHeaders({ 'responseType': 'json' }) }).pipe(
      tap(_ => this.log('pinged')),
      catchError(this.handleError<String>('getPing', ""))
    )
  }

  loadDefaultOCEL() {
    return this.http.get<any>(this.backendUrl + 'load-default-ocel').pipe(
      tap(_ => this.log('ocel posted')),
      catchError(this.handleError<String>('postOCEL', ""))
    )
  }

  postOCEL(formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'upload-ocel', formData).pipe(
      tap(_ => this.log('ocel posted')),
      catchError(this.handleError<String>('postOCEL', ""))
    )
  }

  postOCELConfig(session_key: string, formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'ocel-config?sessionKey=' + session_key, formData).pipe(
      tap(_ => this.log('ocel posted')),
      catchError(this.handleError<String>('postOCEL', ""))
    )
  }

  getObjectModelNames(session_key: string) {
    return this.http.get<any>(this.backendUrl + 'object-model-names'
      + '?sessionKey=' + session_key
    ).pipe(
      tap(_ => this.log('object model names queried')),
      catchError(this.handleError<String>('getObjectModelNames', ""))
    )
  }

  initializeObjectGenerator(session_key: string, formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'initialize-object-generator?sessionKey=' + session_key, formData).pipe(
      tap(_ => this.log('object generator initialized')),
      catchError(this.handleError<String>('initializeObjectGenerator', ""))
    )
  }

  setObjectModelName(session_key: string, name: string) {
    return this.http.get<any>(this.backendUrl + 'name-objects?sessionKey=' + session_key + '&name=' + name).pipe(
      tap(_ => this.log('objects named')),
      catchError(this.handleError<String>('startSimulation', ""))
    )
  }

  postObjectModelGeneration(session_key: string, formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'generate-objects?sessionKey=' + session_key, formData).pipe(
      tap(_ => this.log('object model generation posted')),
      catchError(this.handleError<String>('postObjectModelGeneration', ""))
    )
  }

  getParameters(session_key: string, otype: string, parameter_type: string) {
    let request_parameters = 'sessionKey=' + session_key + "&"
      + 'otype=' + otype + "&"
      + 'parameterType=' + parameter_type
    return this.http.get<any>(this.backendUrl + 'generator-parameters?' + request_parameters
    ).pipe(
      tap(_ => this.log('object stats queried')),
      catchError(this.handleError<String>('getObjectModelStats', ""))
    )
  }

  selectForTraining(session_key: string, otype: string, parameter_type: string, attribute: string, selected: boolean) {
    let request_parameters = 'sessionKey=' + session_key + "&"
      + 'parameterType=' + parameter_type + "&"
      + 'otype=' + otype + "&"
      + 'attribute=' + attribute + "&"
      + 'selected=' + (selected ? "True" : "False")
    return this.http.get<any>(this.backendUrl + 'select-for-training?' + request_parameters
    ).pipe(
      tap(_ => this.log('attribute (un-)selected for training')),
      catchError(this.handleError<String>('selectForTraining', ""))
    )
  }

  switchModel(session_key: string, otype: string, parameter_type: string, attribute: string, fitting_model: string) {
    let request_parameters = 'sessionKey=' + session_key + "&"
      + 'parameterType=' + parameter_type + "&"
      + 'otype=' + otype + "&"
      + 'attribute=' + attribute + "&"
      + 'fittingModel=' + fitting_model
    return this.http.get<any>(this.backendUrl + 'switch-model?' + request_parameters
    ).pipe(
      tap(_ => this.log('fitting model switched')),
      catchError(this.handleError<String>('switchModel', ""))
    )
  }

  changeParameters(session_key: string, parameter_type: string, otype: string, attribute: string, parameters: string) {
    const form_data = new FormData();
    form_data.append("parameterType", parameter_type);
    form_data.append("otype", otype);
    form_data.append("attribute", attribute);
    form_data.append("parameters", parameters);
    return this.http.post<any>(this.backendUrl + 'change-parameters?sessionKey=' + session_key, form_data).pipe(
      tap(_ => this.log('parameter configuration changed')),
      catchError(this.handleError<String>('setParameters', ""))
    )
  }

  getArrivalTimesStats(session_key: string, selected_plot_type: string) {
    return this.http.get<any>(this.backendUrl + 'arrival-times'
      + '?sessionKey=' + session_key + "&"
      + 'otype=' + selected_plot_type
    ).pipe(
      tap(_ => this.log('arrival times stats queried')),
      catchError(this.handleError<String>('getArrivalTimesStats', ""))
    )
  }

  getDefaultParameters(session_key: string, otype: string, stats_key: string, attribute_name: string) {
    return this.http.get<any>(this.backendUrl + 'default-parameters'
      + '?sessionKey=' + session_key + "&"
      + 'statsKey=' + stats_key + "&"
      + 'otype=' + otype + "&"
      + 'attribute=' + attribute_name
    ).pipe(
      tap(_ => this.log('parameters queried')),
      catchError(this.handleError<String>('getObjectModelStats', ""))
    )
  }

  discoverOCPN(session_key: string, form_data: FormData) {
    return this.http.post<any>(this.backendUrl + 'discover-ocpn?sessionKey=' + session_key, form_data).pipe(
      tap(_ => this.log('ocpn discovery task posted')),
      catchError(this.handleError<String>('discoverOCPN', ""))
    )
  }

  getSimulationState(session_key: string) {
    return this.http.get<any>(
      this.backendUrl + 'simulation-state?session_key=' + session_key,
      { headers: new HttpHeaders({ 'responseType': 'json' }) })
      .pipe(
        tap(_ => this.log('simulation state queried')),
        catchError(this.handleError<String>('getSimulationState', ""))
      )
  }

  initializeSimulation(session_key: string, use_original_marking: boolean, object_model_name = "") {
    let useOriginalMarking_str = String(use_original_marking)
    let url = this.backendUrl + 'initialize-simulation?&sessionKey=' + session_key
      + "&useOriginalMarking=" + useOriginalMarking_str
      + "&objectModelName=" + object_model_name
    return this.http.get<any>(url).pipe(
      tap(_ => this.log('simulation initiated')),
      catchError(this.handleError<String>('startSimulation', ""))
    )
  }

  startSimulation(steps: number, session_key: string, use_original_marking: boolean, object_model_name: string = "") {
    let useOriginalMarking_str = String(use_original_marking)
    return this.http.get<any>(this.backendUrl + 'simulate?steps=' + steps
      + "&sessionKey=" + session_key
      + "&useOriginalMarking=" + useOriginalMarking_str
      + "&objectModelName=" + object_model_name
    ).pipe(
      tap(_ => this.log('simulation ran')),
      catchError(this.handleError<String>('startSimulation', ""))
    )
  }

  getAvailableSimulationsObjectModels(sessionKey: string) {
    return this.http.get<any>(this.backendUrl + 'available-simulated-models?sessionKey=' + sessionKey).pipe(
      tap(_ => this.log('simulated models queried')),
      catchError(this.handleError<String>('getAvailableSimulationsObjectModels', ""))
    )
  }

  updateEvaluationSelectedObjectModels(session_key: string, selected_object_models: string[]) {
    const form_data = new FormData();
    form_data.append("selectedObjectModels", JSON.stringify(selected_object_models));
    return this.http.post<any>(this.backendUrl + 'update-evaluation-selected-object-models?sessionKey=' + session_key, form_data).pipe(
      tap(_ => this.log('evaluation selected models updated')),
      catchError(this.handleError<String>('updateEvaluationSelectedObjectModels', ""))
    )
  }


  getEvaluation(session_key: string, stats_type: string, otype: string) {
    return this.http.get<any>(this.backendUrl + "evaluate?sessionKey=" + session_key
      + "&statsType=" + stats_type
      + "&otype=" + otype
    ).pipe(
      tap(_ => this.log('got evaluation')),
      catchError(this.handleError<String>('startSimulation', "")))
  }

  public exportOCEL(session_key: string) {
    return this.http.get(this.backendUrl + 'ocel-export?&sessionKey=' + session_key,
      {
        observe: "response",
        responseType: "blob"
      })
  }

  private log(message: string) {
    this.messageService.add(`HeroService: ${message}`);
  }

  private handleError<T>(operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {
      console.error(error);
      this.log(`${operation} failed: ${error.message}`);
      return of(result as T);
    };
  }

}
