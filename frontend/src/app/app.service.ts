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

  initializeObjectGenerator(session_key: string, formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'initialize-object-generator?sessionKey=' + session_key, formData).pipe(
      tap(_ => this.log('object generator initialized')),
      catchError(this.handleError<String>('initializeObjectGenerator', ""))
    )
  }  

  postObjectModelGeneration(session_key: string, formData: FormData) {
    return this.http.post<any>(this.backendUrl + 'generate-object-model?sessionKey=' + session_key, formData).pipe(
      tap(_ => this.log('object model generation posted')),
      catchError(this.handleError<String>('postObjectModelGeneration', ""))
    )
  }

  getStats(session_key: string, stats_key: string, otype: string) {
    return this.http.get<any>(this.backendUrl + 'object-stats'
      + '?sessionKey=' + session_key + "&"
      + 'statsKey=' + stats_key + "&"
      + 'otype=' + otype
    ).pipe(
      tap(_ => this.log('object stats queried')),
      catchError(this.handleError<String>('getObjectModelStats', ""))
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

  initializeSimulation(session_key: string, useOriginalMarking: boolean) {
    let useOriginalMarking_str = String(useOriginalMarking)
    let url = this.backendUrl + 'initialize-simulation?&sessionKey=' + session_key + "&useOriginalMarking=" + useOriginalMarking_str
    return this.http.get<any>(url).pipe(
      tap(_ => this.log('simulation initiated')),
      catchError(this.handleError<String>('startSimulation', ""))
    )  }

  startSimulation(steps: number, session_key: string) {
    return this.http.get<any>(this.backendUrl + 'simulate?steps=' + steps + "&sessionKey=" + session_key).pipe(
      tap(_ => this.log('simulation ran')),
      catchError(this.handleError<String>('startSimulation', ""))
    )
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
